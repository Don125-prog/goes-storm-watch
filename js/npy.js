/**
 * Minimal .npy file parser for float32 arrays.
 * Supports NumPy format 1.0 with C-order float32/float64 data.
 *
 * Returns { shape: number[], data: Float32Array }
 */
function parseNpy(buffer) {
    const view = new DataView(buffer);

    // Magic: \x93NUMPY
    const magic = String.fromCharCode(
        view.getUint8(0), view.getUint8(1), view.getUint8(2),
        view.getUint8(3), view.getUint8(4), view.getUint8(5)
    );
    if (magic !== '\x93NUMPY') {
        throw new Error('Not a valid .npy file');
    }

    const major = view.getUint8(6);
    // const minor = view.getUint8(7);

    let headerLen;
    let headerOffset;
    if (major === 1) {
        headerLen = view.getUint16(8, true); // little-endian
        headerOffset = 10;
    } else if (major === 2) {
        headerLen = view.getUint32(8, true);
        headerOffset = 12;
    } else {
        throw new Error(`Unsupported npy version: ${major}`);
    }

    const headerStr = new TextDecoder('ascii').decode(
        new Uint8Array(buffer, headerOffset, headerLen)
    );

    // Parse dtype
    const dtypeMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
    if (!dtypeMatch) throw new Error('Cannot parse dtype from header');
    const dtype = dtypeMatch[1];

    // Parse shape
    const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
    if (!shapeMatch) throw new Error('Cannot parse shape from header');
    const shape = shapeMatch[1]
        .split(',')
        .map(s => s.trim())
        .filter(s => s.length > 0)
        .map(Number);

    // Parse fortran_order
    const fortranMatch = headerStr.match(/'fortran_order'\s*:\s*(True|False)/);
    const fortranOrder = fortranMatch && fortranMatch[1] === 'True';
    if (fortranOrder) throw new Error('Fortran order not supported');

    const dataOffset = headerOffset + headerLen;
    const totalElements = shape.reduce((a, b) => a * b, 1);

    let data;
    // Support both little-endian and big-endian float32/float64
    if (dtype === '<f4' || dtype === 'float32') {
        data = new Float32Array(buffer, dataOffset, totalElements);
    } else if (dtype === '<f8' || dtype === 'float64') {
        const f64 = new Float64Array(buffer, dataOffset, totalElements);
        data = new Float32Array(totalElements);
        for (let i = 0; i < totalElements; i++) data[i] = f64[i];
    } else if (dtype === '>f4') {
        // Big-endian float32 — convert manually
        data = new Float32Array(totalElements);
        const dv = new DataView(buffer, dataOffset);
        for (let i = 0; i < totalElements; i++) {
            data[i] = dv.getFloat32(i * 4, false);
        }
    } else {
        throw new Error(`Unsupported dtype: ${dtype}. Expected float32 or float64.`);
    }

    return { shape, data };
}
