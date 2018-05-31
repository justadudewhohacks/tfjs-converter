import * as tfc from '@tensorflow/tfjs-core';
import { NamedTensorsMap } from '../../data/types';
import { ExecutionContext } from '../../executor/execution_context';
import { Node } from '../types';
import { getParamValue } from './utils';
import { tfcPatched, TensorArray } from '../../tfcpatched';

export function executeExperimentalOp(
  node: Node,
  tensorMap: NamedTensorsMap,
  context: ExecutionContext
): tfc.Tensor[]|TensorArray[]|Array<tfc.Tensor|TensorArray> {
  switch (node.op) {
    case 'unpack': {
      const value = getParamValue('value', node, tensorMap, context) as tfc.Tensor;
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      return tfc.unstack(
        value,
        axis
      );
    }
    case 'nonMaxSuppression': {
      const boxes = getParamValue('boxes', node, tensorMap, context) as tfc.Tensor2D;
      const scores = getParamValue('scores', node, tensorMap, context) as tfc.Tensor1D;
      const maxOutputSize = getParamValue('maxOutputSize', node, tensorMap, context) as tfc.Tensor1D;
      const iouThreshold = getParamValue('iouThreshold', node, tensorMap, context) as number;
      const scoreThreshold = getParamValue('scoreThreshold', node, tensorMap, context) as number;

      return [
        tfcPatched.nonMaxSuppression(
          boxes,
          scores,
          maxOutputSize,
          iouThreshold,
          scoreThreshold
        )
      ];
    }
    case 'tensorArray': {
      const size = getParamValue('size', node, tensorMap, context) as tfc.Tensor1D;
      const dtype = getParamValue('dtype', node, tensorMap, context) as string;
      const elementShape = getParamValue('elementShape', node, tensorMap, context) as number[];
      const dynamicSize = getParamValue('dynamicSize', node, tensorMap, context) as boolean;
      const clearAfterRead = getParamValue('clearAfterRead', node, tensorMap, context) as boolean;
      const identicalElementShapes = getParamValue('identicalElementShapes', node, tensorMap, context) as boolean;
      const tensorArrayName = getParamValue('tensorArrayName', node, tensorMap, context) as string;
      return [
        tfcPatched.tensorArray(
          size,
          dtype,
          elementShape,
          dynamicSize,
          clearAfterRead,
          identicalElementShapes,
          tensorArrayName
        )
      ];
    }
    default:
      throw TypeError(`experimental Node type ${node.op} is not implemented`);
  }
}
