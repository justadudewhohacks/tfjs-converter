import * as tfc from '@tensorflow/tfjs-core';

const { util } = tfc;

export function nonMaxSuppression(
  boxes: tfc.Tensor2D,
  scores: tfc.Tensor1D,
  maxOutputSize: tfc.Tensor1D,
  iouThreshold: number,
  scoreThreshold: number
): tfc.Tensor1D {
  util.assert(
    0 <= iouThreshold && iouThreshold <= 1,
    'iouThreshold must be in [0, 1]');
  util.assert(
    boxes.shape.length === 2,
    'boxes must be 2-D');
  util.assert(
    boxes.shape[1] === 4,
    'boxes must have 4 columns');

  const numBoxes = boxes.shape[0];
  util.assert(
    scores.shape.length === 1,
    'scores must be 1-D');
  util.assert(
    scores.shape[0] === numBoxes,
    'scores has incompatible shape, expected [numBoxes]');

  const outputSize = Math.min(
    ...Array.from(maxOutputSize.dataSync()),
    numBoxes
  );

  const candidates = Array.from(scores.dataSync())
    .map((score, boxIndex) => ({ score, boxIndex }))
    .filter(c => c.score > scoreThreshold)
    .sort((c1, c2) => c2.score - c1.score);

  const suppressFunc = (x: number) => x <= iouThreshold ? 1 : 0;

  const selected: number[] = [];

  candidates.forEach(c => {
    if (selected.length >= outputSize) {
      return;
    }
    const originalScore = c.score;

    for (let j = selected.length - 1; j >= 0; --j) {
      const iou = IOU(boxes, c.boxIndex, selected[j]);
      if (iou === 0.0) {
        continue;
      }
      c.score *= suppressFunc(iou);
      if (c.score <= scoreThreshold) {
        break;
      }
    }

    if (originalScore === c.score) {
      selected.push(c.boxIndex);
    }
  });

  return tfc.tensor1d(selected, 'int32');
}

function IOU(boxes: tfc.Tensor2D, i: number, j: number) {
  const yminI = Math.min(boxes.get(i, 0), boxes.get(i, 2));
  const xminI = Math.min(boxes.get(i, 1), boxes.get(i, 3));
  const ymaxI = Math.max(boxes.get(i, 0), boxes.get(i, 2));
  const xmaxI = Math.max(boxes.get(i, 1), boxes.get(i, 3));
  const yminJ = Math.min(boxes.get(j, 0), boxes.get(j, 2));
  const xminJ = Math.min(boxes.get(j, 1), boxes.get(j, 3));
  const ymaxJ = Math.max(boxes.get(j, 0), boxes.get(j, 2));
  const xmaxJ = Math.max(boxes.get(j, 1), boxes.get(j, 3));
  const areaI = (ymaxI - yminI) * (xmaxI - xminI);
  const areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
  if (areaI <= 0 || areaJ <= 0) {
    return 0.0;
  }
  const intersectionYmin = Math.max(yminI, yminJ);
  const intersectionXmin = Math.max(xminI, xminJ);
  const intersectionYmax = Math.min(ymaxI, ymaxJ);
  const intersectionXmax = Math.min(xmaxI, xmaxJ);
  const intersectionArea =
      Math.max(intersectionYmax - intersectionYmin, 0.0) *
      Math.max(intersectionXmax - intersectionXmin, 0.0);
  return intersectionArea / (areaI + areaJ - intersectionArea);
}
