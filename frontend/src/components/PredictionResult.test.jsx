import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import PredictionResult from './PredictionResult.jsx';

const sampleResult = {
  prediction: 'nv',
  deterministic_prediction: 'nv',
  confidence: 0.76,
  uncertainty: 0.82,
  uncertainty_normalized: 0.42,
  mc_passes: 30,
  probabilities: { akiec: 0.02, bcc: 0.04, bkl: 0.08, df: 0.01, mel: 0.05, nv: 0.76, vasc: 0.04 },
  class_probability_variance: {},
  inference_time_ms: 810,
  model_name: 'EfficientNet-B3',
  model_version: 'test-model',
  device: 'cpu',
};

describe('PredictionResult', () => {
  it('renders the model result and uncertainty disclaimer', () => {
    render(<PredictionResult result={sampleResult} />);
    expect(screen.getByRole('heading', { name: 'nv' })).toBeInTheDocument();
    expect(screen.getByText('76.0%')).toBeInTheDocument();
    expect(screen.getByText(/not medical severity/i)).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /model probability distribution/i })).toBeInTheDocument();
  });
});

