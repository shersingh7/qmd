/**
 * protocol.ts — Binary & JSON Wire Protocol Serialization and Validation
 *
 * Implements strict binary framing and JSON response validation for high-speed,
 * safe embedding transport over local loopback HTTP.
 */

export class EmbeddingProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EmbeddingProtocolError";
  }
}

export interface DecodedBinaryPayload {
  count: number;
  dims: number;
  vectors: number[][];
  rawFloats: Float32Array;
}

/**
 * Decodes little-endian binary embedding payload.
 * Binary format: [int32 count (4B)][int32 dims (4B)][float32 data (count * dims * 4B)]
 */
export function decodeBinaryEmbeddings(buffer: ArrayBuffer): DecodedBinaryPayload {
  if (buffer.byteLength < 8) {
    throw new EmbeddingProtocolError(
      `Binary response truncated: buffer length ${buffer.byteLength} < 8-byte header`
    );
  }

  // Use DataView to ensure strictly little-endian parsing
  const view = new DataView(buffer);
  const count = view.getInt32(0, true);
  const dims = view.getInt32(4, true);

  if (count < 0 || dims < 0) {
    throw new EmbeddingProtocolError(`Invalid negative count (${count}) or dims (${dims}) in header`);
  }

  if (count === 0 || dims === 0) {
    return { count: 0, dims: 0, vectors: [], rawFloats: new Float32Array(0) };
  }

  const expectedDataBytes = count * dims * 4;
  const totalExpectedBytes = 8 + expectedDataBytes;

  if (buffer.byteLength !== totalExpectedBytes) {
    throw new EmbeddingProtocolError(
      `Binary payload byte length mismatch: expected ${totalExpectedBytes} bytes for ${count}x${dims} floats, received ${buffer.byteLength} bytes`
    );
  }

  // Float32Array view directly into buffer offset 8
  const floats = new Float32Array(buffer, 8, count * dims);

  // Validate finite values (no NaN or Infinity)
  for (let i = 0; i < floats.length; i++) {
    const val = floats[i]!;
    if (!Number.isFinite(val)) {
      throw new EmbeddingProtocolError(`Non-finite float value detected in embedding at offset ${i}: ${val}`);
    }
  }

  const vectors: number[][] = new Array(count);
  for (let i = 0; i < count; i++) {
    const start = i * dims;
    const end = start + dims;
    vectors[i] = Array.from(floats.subarray(start, end));
  }

  return {
    count,
    dims,
    vectors,
    rawFloats: floats,
  };
}

/**
 * Encodes count x dims float32 vectors into standard little-endian binary buffer.
 */
export function encodeBinaryEmbeddings(vectors: number[][] | Float32Array[], dims: number): Uint8Array {
  const count = vectors.length;
  const totalBytes = 8 + count * dims * 4;
  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);

  view.setInt32(0, count, true);
  view.setInt32(4, dims, true);

  const floatView = new Float32Array(buffer, 8, count * dims);
  for (let i = 0; i < count; i++) {
    const vec = vectors[i]!;
    if (vec.length !== dims) {
      throw new EmbeddingProtocolError(`Vector at index ${i} has length ${vec.length}, expected ${dims}`);
    }
    for (let j = 0; j < dims; j++) {
      const v = vec[j]!;
      if (!Number.isFinite(v)) {
        throw new EmbeddingProtocolError(`Non-finite float value at vector ${i}, dim ${j}: ${v}`);
      }
      floatView[i * dims + j] = v;
    }
  }

  return new Uint8Array(buffer);
}

/**
 * Validates a JSON response payload from the embedding server.
 */
export function validateJsonEmbedResponse(data: unknown): {
  embeddings: number[][];
  model: string;
  dims: number;
} {
  if (!data || typeof data !== "object") {
    throw new EmbeddingProtocolError("Invalid JSON response: expected an object");
  }

  const obj = data as Record<string, unknown>;
  if (!Array.isArray(obj.embeddings)) {
    throw new EmbeddingProtocolError("Invalid JSON response: 'embeddings' must be an array");
  }

  const embeddings = obj.embeddings as number[][];
  const model = typeof obj.model === "string" ? obj.model : "unknown";
  const reportedDims = typeof obj.dims === "number" ? obj.dims : (embeddings[0]?.length ?? 0);

  for (let i = 0; i < embeddings.length; i++) {
    const vec = embeddings[i];
    if (!Array.isArray(vec)) {
      throw new EmbeddingProtocolError(`Embedding at index ${i} is not an array`);
    }
    if (reportedDims > 0 && vec.length !== reportedDims) {
      throw new EmbeddingProtocolError(
        `Embedding at index ${i} length ${vec.length} does not match expected dims ${reportedDims}`
      );
    }
    for (let j = 0; j < vec.length; j++) {
      const val = vec[j];
      if (typeof val !== "number" || !Number.isFinite(val)) {
        throw new EmbeddingProtocolError(`Non-finite number at embedding ${i}, dim ${j}: ${val}`);
      }
    }
  }

  return {
    embeddings,
    model,
    dims: reportedDims,
  };
}
