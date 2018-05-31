import * as tfc from '@tensorflow/tfjs-core';

export class TensorArray {
  constructor(
    public size: tfc.Tensor1D,
    public dtype: string,
    public elementShape: number[],
    public dynamicSize: boolean,
    public clearAfterRead: boolean,
    public identicalElementShapes: boolean,
    public tensorArrayName: string
  ) {}
}
