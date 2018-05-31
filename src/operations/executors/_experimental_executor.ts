import * as tfc from '@tensorflow/tfjs-core';
import { NamedTensorsMap } from '../../data/types';
import { ExecutionContext } from '../../executor/execution_context';
import { Node } from '../types';
import { getParamValue } from './utils';
import { tfcPatched } from '../../tfcpatched/index';

export function executeExperimentalOp(
  node: Node,
  tensorMap: NamedTensorsMap,
  context: ExecutionContext
): tfc.Tensor[] {
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
    default:
      throw TypeError(`experimental Node type ${node.op} is not implemented`);
  }
}
