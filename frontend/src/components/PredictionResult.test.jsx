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

const melanomaResult = {
  prediction: 'mel',
  deterministic_prediction: 'mel',
  confidence: 0.89,
  uncertainty: 0.35,
  uncertainty_normalized: 0.18,
  mc_passes: 30,
  probabilities: { akiec: 0.01, bcc: 0.03, bkl: 0.02, df: 0.01, mel: 0.89, nv: 0.03, vasc: 0.01 },
  class_probability_variance: {},
  inference_time_ms: 750,
  model_name: 'EfficientNet-B3',
  model_version: 'test-model',
  device: 'cpu',
};

describe('PredictionResult', () => {
  it('renders the human-readable predicted class, badges, and clinical guide for benign nevus', () => {
    render(<PredictionResult result={sampleResult} />);

    // Friendly class heading and code
    expect(screen.getByRole('heading', { name: /Melanocytic Nevus/i })).toBeInTheDocument();
    expect(screen.getByText('Benign (Harmless)')).toBeInTheDocument();
    expect(screen.getByText('Code: nv')).toBeInTheDocument();

    // Metrics
    expect(screen.getByText('76.0%')).toBeInTheDocument();
    expect(screen.getByText(/Common Mole \(nv\)/i)).toBeInTheDocument();
    expect(screen.getByText(/not medical severity/i)).toBeInTheDocument();

    // Educational Clinical Guide card
    expect(screen.getByRole('heading', { name: /Understanding Common Mole/i })).toBeInTheDocument();
    expect(screen.getByText(/clustered pigment-producing cells/i)).toBeInTheDocument();
    expect(screen.getByText('Pathological Nature')).toBeInTheDocument();
    expect(screen.getByText('Visual Characteristics')).toBeInTheDocument();
    expect(screen.getByText('Recommended Next Steps')).toBeInTheDocument();

    // Probability Chart
    expect(screen.getByRole('img', { name: /model probability distribution/i })).toBeInTheDocument();
  });

  it('renders high-risk malignant labels and urgent guidance when melanoma is predicted', () => {
    render(<PredictionResult result={melanomaResult} />);

    expect(screen.getByRole('heading', { name: /Melanoma \(Malignant Skin Cancer\)/i })).toBeInTheDocument();
    expect(screen.getByText('Malignant (High Urgency)')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /Understanding Melanoma/i })).toBeInTheDocument();
    expect(screen.getByText('Critical / Urgent (Malignant)')).toBeInTheDocument();
    expect(screen.getByText(/ABCDE criteria/i)).toBeInTheDocument();
    expect(screen.getByText(/Urgent clinical evaluation required/i)).toBeInTheDocument();
  });
});
