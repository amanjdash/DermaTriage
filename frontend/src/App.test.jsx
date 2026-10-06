import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App.jsx';
import { analyzeImage, getHealth } from './services/api.js';

vi.mock('./services/api.js', () => ({
  getHealth: vi.fn(),
  analyzeImage: vi.fn(),
}));

const result = {
  prediction: 'nv', deterministic_prediction: 'nv', confidence: 0.8,
  uncertainty: 0.4, uncertainty_normalized: 0.2, mc_passes: 30,
  probabilities: { akiec: 0.01, bcc: 0.02, bkl: 0.03, df: 0.01, mel: 0.08, nv: 0.8, vasc: 0.05 },
  class_probability_variance: {}, inference_time_ms: 30, model_name: 'EfficientNet-B3',
  model_version: 'test-model', device: 'cpu',
};

beforeEach(() => {
  getHealth.mockResolvedValue({ status: 'ok', model_loaded: true });
  analyzeImage.mockReset();
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:preview'), revokeObjectURL: vi.fn() });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('dashboard flow', () => {
  it('shows a preview, loading state, and successful results', async () => {
    const user = userEvent.setup();
    let finishAnalysis;
    analyzeImage.mockReturnValue(new Promise((resolve) => { finishAnalysis = resolve; }));
    render(<App />);
    const input = screen.getByLabelText('Choose a dermoscopic image');
    await user.upload(input, new File(['image bytes'], 'lesion.png', { type: 'image/png' }));
    expect(await screen.findByAltText('Selected lesion image preview')).toHaveAttribute('src', 'blob:preview');
    await user.click(screen.getByRole('button', { name: /analyze image/i }));
    expect(screen.getByRole('button', { name: /analyzing image/i })).toBeDisabled();
    finishAnalysis(result);
    expect(await screen.findByRole('heading', { name: 'Analysis summary' })).toBeInTheDocument();
    expect(screen.getByText('80.0%')).toBeInTheDocument();
  });

  it('shows an understandable API error', async () => {
    const user = userEvent.setup();
    analyzeImage.mockRejectedValue(new Error('The DermaTriage API could not be reached. Check that the backend is running.'));
    render(<App />);
    await user.upload(screen.getByLabelText('Choose a dermoscopic image'), new File(['x'], 'lesion.jpg', { type: 'image/jpeg' }));
    await user.click(screen.getByRole('button', { name: /analyze image/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent(/API could not be reached/i);
  });
});

