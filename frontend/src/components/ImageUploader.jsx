import { useRef, useState } from 'react';

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export function validateImage(file) {
  if (!file) return 'Choose an image to continue.';
  if (!ACCEPTED_TYPES.includes(file.type)) return 'Choose a JPEG, PNG, or WebP image.';
  if (file.size > MAX_UPLOAD_BYTES) return 'This image is larger than the 10 MB limit.';
  return null;
}

export default function ImageUploader({ file, previewUrl, onFileChange, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState('');

  function chooseFile(nextFile) {
    const validationError = validateImage(nextFile);
    setError(validationError || '');
    if (!validationError) onFileChange(nextFile);
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  }

  return (
    <section className="upload-card card" aria-labelledby="upload-title">
      <div className="section-heading">
        <div><span className="eyebrow">01 / IMAGE</span><h2 id="upload-title">Add a lesion image</h2></div>
        <span className="step-number">01</span>
      </div>
      <div
        className={`dropzone ${dragging ? 'dropzone--active' : ''} ${file ? 'dropzone--has-file' : ''}`}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }}
        onDrop={onDrop}
      >
        {file ? (
          <>
            <img className="image-preview" src={previewUrl} alt="Selected lesion image preview" />
            <div className="file-details"><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · ready for analysis</span></div>
            <button className="text-button" type="button" disabled={disabled} onClick={() => inputRef.current?.click()}>Replace image</button>
          </>
        ) : (
          <>
            <div className="upload-icon" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none"><path d="M16 21V6m0 0L10 12M16 6l6 6M7 20v5a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg></div>
            <p className="drop-title">Drop an image here</p>
            <p className="drop-subtitle">or choose a file from your device</p>
            <button className="button button--quiet" type="button" disabled={disabled} onClick={() => inputRef.current?.click()}>Choose image</button>
            <p className="file-hint">JPEG, PNG or WebP · up to 10 MB</p>
          </>
        )}
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          aria-label="Choose a dermoscopic image"
          disabled={disabled}
          onChange={(event) => chooseFile(event.target.files?.[0])}
        />
      </div>
      {error && <p className="field-error" role="alert">{error}</p>}
      <div className="privacy-note"><span aria-hidden="true">⌁</span><p>Images are used for this request and are not retained by the application.</p></div>
    </section>
  );
}

