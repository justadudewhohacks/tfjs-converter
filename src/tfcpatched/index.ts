import { loadWeights } from './loadWeights';
import { nonMaxSuppression } from './ops/nonMaxSuppression';
import { tensorArray } from './ops/tensorArray';

export { TensorArray } from './TensorArray';

export const tfcPatched = {
  loadWeights,
  nonMaxSuppression,
  tensorArray
};
