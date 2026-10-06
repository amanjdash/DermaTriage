import { describe, expect, it } from 'vitest';
import { validateImage, MAX_UPLOAD_BYTES } from './ImageUploader.jsx';

describe('validateImage', () => {
  it('accepts JPEG, PNG and WebP within the size limit', () => {
    expect(validateImage(new File(['x'], 'lesion.jpg', { type: 'image/jpeg' }))).toBeNull();
    expect(validateImage(new File(['x'], 'lesion.png', { type: 'image/png' }))).toBeNull();
    expect(validateImage(new File(['x'], 'lesion.webp', { type: 'image/webp' }))).toBeNull();
  });

  it('rejects unsupported media and oversized files', () => {
    expect(validateImage(new File(['x'], 'script.svg', { type: 'image/svg+xml' }))).toMatch(/JPEG, PNG, or WebP/);
    expect(validateImage(new File([new Uint8Array(MAX_UPLOAD_BYTES + 1)], 'large.jpg', { type: 'image/jpeg' }))).toMatch(/10 MB/);
  });
});

