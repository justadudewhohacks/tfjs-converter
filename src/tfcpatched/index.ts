import { loadWeights } from './loadWeights';
import { nonMaxSuppression } from './ops/nonMaxSuppression';

export const tfcPatched = {
  loadWeights,
  nonMaxSuppression
};
