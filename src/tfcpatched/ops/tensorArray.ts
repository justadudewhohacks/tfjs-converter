import * as tfc from '@tensorflow/tfjs-core';

import { TensorArray } from '../TensorArray';

export function tensorArray(
  size: tfc.Tensor1D,
  dtype: string,
  elementShape: number[],
  dynamicSize: boolean,
  clearAfterRead: boolean,
  identicalElementShapes: boolean,
  tensorArrayName: string
): TensorArray {
  return new TensorArray(
    size, dtype, elementShape, dynamicSize, clearAfterRead, identicalElementShapes, tensorArrayName
  );
}
