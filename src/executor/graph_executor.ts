/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { NamedTensorMap, Tensor, tidy } from '@tensorflow/tfjs-core';

import { NamedTensorsMap } from '../data/types';
import { executeExperimentalOp } from '../operations/executors/_experimental_executor';
import { getNodeNameAndIndex, getTensor, getTensorOrTensorArray } from '../operations/executors/utils';
import { executeOp as _executeOp } from '../operations/operation_executor';
import { Graph, Node } from '../operations/types';
import { ExecutionContext, ExecutionContextInfo } from './execution_context';
import { TensorArray } from '../tfcpatched';

function isExperimentalOp(node: Node): boolean {
  const experimentalOps = [
    'unpack',
    'nonMaxSuppression',
    'tensorArray'
  ];
  return experimentalOps.some(op => op === node.op);
}

function executeOp(
  node: Node,
  tensorMap: NamedTensorsMap,
  context: ExecutionContext
): Array<Tensor|TensorArray>|Promise<Array<Tensor|TensorArray>> {

  if (node.op === 'const') {
    return tensorMap[node.name];
  }

  if (isExperimentalOp(node)) {
    return executeExperimentalOp(node, tensorMap, context);
  }
  return _executeOp(node, tensorMap, context);
}

interface NodeWithContexts {
  contexts: ExecutionContextInfo[];
  node: Node;
}

export class GraphExecutor {
  private compiledOrder: Node[] = [];
  private _weightMap: NamedTensorsMap = {};
  private weightIds: number[];
  private placeholders: string[];
  private outputs: string[];
  get weightMap(): NamedTensorsMap {
    return this._weightMap;
  }
  set weightMap(weightMap: NamedTensorsMap) {
    const weightIds = Object.keys(weightMap).map(
        key => weightMap[key].filter(el => el instanceof Tensor).map((tensor: Tensor) => tensor.id)
      );
    this.weightIds = [].concat.apply([], weightIds);
    this._weightMap = weightMap;
  }

  get inputNodes(): string[] {
    return this.placeholders;
  }

  get outputNodes(): string[] {
    return this.outputs;
  }

  constructor(private graph: Graph) {
    this.placeholders = graph.placeholders.map(node => node.name);
    this.outputs = graph.outputs.map(node => node.name);
    this.compile();
  }

  get isControlFlowModel(): boolean {
    return this.graph.withControlFlow;
  }

  /**
   * Compiles the inference graph to generate the topology order of op nodes,
   * cache the result for inference execution.
   */
  private compile() {
    // Do not compile for graph with control flow, since the execution order
    // requires runtime evaluation of the output tensors.
    if (this.graph.withControlFlow) {
      return;
    }

    const stack = [...this.graph.inputs];
    const visited: {[key: string]: boolean} = {};
    while (stack.length > 0) {
      const node = stack.pop();
      visited[node.name] = true;
      this.compiledOrder.push(node);
      node.children.forEach((childNode) => {
        if (!visited[childNode.name] && childNode.inputNames.every(name => {
              const [nodeName, ] = getNodeNameAndIndex(name);
              return visited[nodeName];
            })) {
          stack.push(childNode);
        }
      });
    }
  }

  /**
   * Executes the inference for given input tensors.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs output node name from the Tensorflow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   */
  execute(inputs: NamedTensorsMap, outputs?: string|string[]): NamedTensorMap {
    this.checkInput(inputs);
    const result = tidy(() => {
      const context = new ExecutionContext(this._weightMap);
      const tensors =
          this.compiledOrder.reduce<NamedTensorsMap>((map, node) => {
            map[node.name] = executeOp(node, map, context) as Tensor[];
            return map;
          }, {...this.weightMap, ...inputs});
      return this.findOutputs(tensors, context, outputs);
    });
    return result;
  }

  /**
   * Executes the inference for given input tensors in Async fashion.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs output node name from the Tensorflow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   */
  async executeAsync(inputs: NamedTensorsMap, outputs?: string|string[]):
      Promise<NamedTensorMap> {
    const context = new ExecutionContext(this._weightMap);
    // Graph with control flow op requires runtime evaluation of the execution
    // order, while without control flow the execution order is pre-determined
    // in the compile method.
    const tensors = await this.executeWithControlFlow(inputs, context);
    const results = this.findOutputs(tensors, context, outputs);

    // dispose all the intermediate tensors
    const outputIds = Object.keys(results)
      .map(key => results[key])
      .filter(el => el instanceof Tensor)
      .map(tensor => tensor.id);
    const inputIdArray =
        Object.keys(inputs)
          .map(key => inputs[key]
          .filter(el => el instanceof Tensor)
          .map((tensor: Tensor) => tensor.id));
    const inputIds = [].concat.apply([], inputIdArray);
    Object.keys(tensors).forEach(key => {
      const tensorArray = tensors[key];
      tensorArray
        .filter(el => el instanceof Tensor)
        .forEach((tensor: Tensor) => {
          if (tensor && outputIds.indexOf(tensor.id) === -1 &&
              inputIds.indexOf(tensor.id) === -1 &&
              this.weightIds.indexOf(tensor.id) === -1) {
            tensor.dispose();
          }
        });
    });
    return results;
  }

  /**
   * When there are control flow nodes in the graph, the graph execution use
   * ExecutionContext to keep track of the frames and loop iterators.
   * @param inputs placeholder tensors for the graph.
   * @param context the execution context object for current execution.
   */
  private async executeWithControlFlow(
      inputs: NamedTensorsMap,
      context: ExecutionContext): Promise<NamedTensorsMap> {
    const stack: NodeWithContexts[] = this.graph.inputs.map(node => {
      return {node, contexts: context.currentContext};
    });
    const tensorMap = {...this.weightMap, ...inputs};
    const added: {[key: string]: boolean} = {};

    while (stack.length > 0) {
      const item = stack.pop();
      context.currentContext = item.contexts;

      const tensors = executeOp(item.node, tensorMap, context);

      const [nodeName, ] = getNodeNameAndIndex(item.node.name, context);
      tensorMap[nodeName] = await tensors;
      item.node.children.forEach((childNode) => {
        const [nodeName, ] = getNodeNameAndIndex(childNode.name, context);
        if (!added[nodeName]) {
          // Merge op can be pushed if any of its inputs has value.
          if (childNode.op === 'merge') {
            if (childNode.inputNames.some(name => {
                  return !!getTensorOrTensorArray(name, tensorMap, context);
                })) {
              added[nodeName] = true;
              stack.push({contexts: context.currentContext, node: childNode});
            }
          } else  // Otherwise all inputs must to have value.
              if (childNode.inputNames.every(name => {
                    return !!getTensorOrTensorArray(name, tensorMap, context);
                  })) {
            added[nodeName] = true;
            stack.push({contexts: context.currentContext, node: childNode});
          }
        }
      });
    }

    return tensorMap;
  }

  private findOutputs(
      tensorMap: NamedTensorsMap, context: ExecutionContext,
      outputs?: string|string[]): NamedTensorMap {
    if (outputs && !(outputs instanceof Array)) {
      outputs = [outputs];
    }
    const requestedOutputs =
        (outputs || this.graph.outputs.map(node => node.name)) as string[];

    return requestedOutputs.reduce<NamedTensorMap>((map, name) => {
      map[name] = getTensor(name, tensorMap, context);
      return map;
    }, {});
  }
  /**
   * Releases the memory used by the weight tensors.
   */
  dispose() {
    Object.keys(this.weightMap)
        .forEach(
            key => this.weightMap[key].forEach(tensor => {
              if (tensor instanceof Tensor) {
                tensor.dispose();
              }
            }));
  }

  private checkInput(inputs: NamedTensorsMap) {
    const inputKeys = Object.keys(inputs);
    const missing: string[] = [];
    const extra: string[] = [];

    this.placeholders.forEach(name => {
      if (inputKeys.indexOf(name) === -1) missing.push(name);
    });

    inputKeys.forEach(name => {
      if (this.placeholders.indexOf(name) === -1) extra.push(name);
    });

    if (missing.length > 0) {
      throw new Error(
          `The dict provided in model.execute(dict) has the keys ` +
          `[${inputKeys}], but is missing the required keys: [${missing}].`);
    }

    if (extra.length > 0) {
      throw new Error(
          `The dict provided in model.execute(dict) has ` +
          `unused keys: [${extra}]. Please provide only the following keys: ` +
          `[${this.placeholders}].`);
    }
  }
}
