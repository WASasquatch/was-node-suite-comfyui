import { Box3, Matrix4, BufferAttribute, Vector3, Vector2, Plane, Line3, Triangle, REVISION, BackSide, DoubleSide, Ray, FrontSide, BufferGeometry, Sphere, Mesh, Object3D, Group, LineBasicMaterial, MeshBasicMaterial, Line, LineSegments, LineLoop, Points, BatchedMesh, RGBAFormat, RGFormat, RedFormat, RGBAIntegerFormat, RGIntegerFormat, RedIntegerFormat, DataTexture, NearestFilter, IntType, UnsignedIntType, FloatType, UnsignedByteType, UnsignedShortType, ByteType, ShortType, Vector4, Matrix3 } from '../three/three.module.mjs';

// Split strategy constants

/**
 * Option for splitting each BVH node down the center of the longest axis of the bounds.
 *
 * This is the fastest construction option and will yield a good, performant bounds.
 * @type {number}
 */
const CENTER = 0;

/**
 * Option for splitting each BVH node at the average point along the longest axis for
 * all triangle centroids in the bounds.
 *
 * This strategy may be better than `CENTER` with some geometry.
 * @type {number}
 */
const AVERAGE = 1;

/**
 * Option to use a Surface Area Heuristic to split the bounds more optimally. This SAH
 * implementation tests 32 discrete splits in each node along each axis to determine
 * which split is the lowest cost.
 *
 * This is the slowest construction option but will yield the best bounds of the three
 * options and use the least memory.
 * @type {number}
 */
const SAH = 2;

// Traversal constants

/**
 * Indicates the shape did not intersect the given bounding box.
 * @type {number}
 */
const NOT_INTERSECTED = 0;

/**
 * Indicates the shape did intersect the given bounding box.
 * @type {number}
 */
const INTERSECTED = 1;

/**
 * Indicate the shape entirely contains the given bounding box.
 * @type {number}
 */
const CONTAINED = 2;

// SAH cost constants
// TODO: hone these costs more. The relative difference between them should be the
// difference in measured time to perform a primitive intersection vs traversing
// bounds.
// TODO: could be tuned per primitive type (triangles vs lines vs points)
const PRIMITIVE_INTERSECT_COST = 1.25;
const TRAVERSAL_COST = 1;


// Build constants
const BYTES_PER_NODE = 6 * 4 + 4 + 4;
const UINT32_PER_NODE = BYTES_PER_NODE / 4;
const IS_LEAFNODE_FLAG = 0xFFFF;

// Bit masks for 32 bit node data
const LEAFNODE_MASK_32 = IS_LEAFNODE_FLAG << 16;

// EPSILON for computing floating point error during build
// https://en.wikipedia.org/wiki/Machine_epsilon#Values_for_standard_hardware_floating_point_arithmetics
const FLOAT32_EPSILON = Math.pow( 2, - 24 );

const SKIP_GENERATION = Symbol( 'SKIP_GENERATION' );

const DEFAULT_OPTIONS = {
	strategy: CENTER,
	maxDepth: 40,
	targetLeafSize: 10,
	useSharedArrayBuffer: false,
	setBoundingBox: true,
	onProgress: null,
	indirect: false,
	verbose: true,
	range: null,
	[ SKIP_GENERATION ]: false,
};

function arrayToBox( nodeIndex32, array, target ) {

	target.min.x = array[ nodeIndex32 ];
	target.min.y = array[ nodeIndex32 + 1 ];
	target.min.z = array[ nodeIndex32 + 2 ];

	target.max.x = array[ nodeIndex32 + 3 ];
	target.max.y = array[ nodeIndex32 + 4 ];
	target.max.z = array[ nodeIndex32 + 5 ];

	return target;

}

function makeEmptyBounds( target ) {

	target[ 0 ] = target[ 1 ] = target[ 2 ] = Infinity;
	target[ 3 ] = target[ 4 ] = target[ 5 ] = - Infinity;

}

function getLongestEdgeIndex( bounds ) {

	let splitDimIdx = - 1;
	let splitDist = - Infinity;

	for ( let i = 0; i < 3; i ++ ) {

		const dist = bounds[ i + 3 ] - bounds[ i ];
		if ( dist > splitDist ) {

			splitDist = dist;
			splitDimIdx = i;

		}

	}

	return splitDimIdx;

}

// copies bounds a into bounds b
function copyBounds( source, target ) {

	target.set( source );

}

// sets bounds target to the union of bounds a and b
function unionBounds( a, b, target ) {

	let aVal, bVal;
	for ( let d = 0; d < 3; d ++ ) {

		const d3 = d + 3;

		// set the minimum values
		aVal = a[ d ];
		bVal = b[ d ];
		target[ d ] = aVal < bVal ? aVal : bVal;

		// set the max values
		aVal = a[ d3 ];
		bVal = b[ d3 ];
		target[ d3 ] = aVal > bVal ? aVal : bVal;

	}

}

// expands the given bounds by the provided primitive bounds
function expandByPrimitiveBounds( startIndex, primitiveBounds, bounds ) {

	for ( let d = 0; d < 3; d ++ ) {

		const tCenter = primitiveBounds[ startIndex + 2 * d ];
		const tHalf = primitiveBounds[ startIndex + 2 * d + 1 ];

		const tMin = tCenter - tHalf;
		const tMax = tCenter + tHalf;

		if ( tMin < bounds[ d ] ) {

			bounds[ d ] = tMin;

		}

		if ( tMax > bounds[ d + 3 ] ) {

			bounds[ d + 3 ] = tMax;

		}

	}

}

// compute bounds surface area
function computeSurfaceArea( bounds ) {

	const d0 = bounds[ 3 ] - bounds[ 0 ];
	const d1 = bounds[ 4 ] - bounds[ 1 ];
	const d2 = bounds[ 5 ] - bounds[ 2 ];

	return 2 * ( d0 * d1 + d1 * d2 + d2 * d0 );

}

function IS_LEAF( n16, uint16Array ) {

	return uint16Array[ n16 + 15 ] === IS_LEAFNODE_FLAG;

}

function OFFSET( n32, uint32Array ) {

	return uint32Array[ n32 + 6 ];

}

function COUNT( n16, uint16Array ) {

	return uint16Array[ n16 + 14 ];

}

// Returns the uint32-aligned offset of the left child node for performance
function LEFT_NODE( n32 ) {

	return n32 + UINT32_PER_NODE;

}

// Returns the uint32-aligned offset of the right child node for performance
function RIGHT_NODE( n32, uint32Array ) {

	// stored value is relative offset from parent, convert to absolute uint32 index
	const relativeOffset = uint32Array[ n32 + 6 ];
	return n32 + relativeOffset * UINT32_PER_NODE;

}

function SPLIT_AXIS( n32, uint32Array ) {

	return uint32Array[ n32 + 7 ];

}

function BOUNDING_DATA_INDEX( n32 ) {

	return n32;

}

// computes the union of the bounds of all of the given primitives and puts the resulting box in "target".
// A bounding box is computed for the centroids of the primitives, as well, and placed in "centroidTarget".
// These are computed together to avoid redundant accesses to bounds array.
function getBounds( primitiveBounds, offset, count, target, centroidTarget ) {

	let minx = Infinity;
	let miny = Infinity;
	let minz = Infinity;
	let maxx = - Infinity;
	let maxy = - Infinity;
	let maxz = - Infinity;

	let cminx = Infinity;
	let cminy = Infinity;
	let cminz = Infinity;
	let cmaxx = - Infinity;
	let cmaxy = - Infinity;
	let cmaxz = - Infinity;

	const boundsOffset = primitiveBounds.offset || 0;
	for ( let i = ( offset - boundsOffset ) * 6, end = ( offset + count - boundsOffset ) * 6; i < end; i += 6 ) {

		const cx = primitiveBounds[ i + 0 ];
		const hx = primitiveBounds[ i + 1 ];
		const lx = cx - hx;
		const rx = cx + hx;
		if ( lx < minx ) minx = lx;
		if ( rx > maxx ) maxx = rx;
		if ( cx < cminx ) cminx = cx;
		if ( cx > cmaxx ) cmaxx = cx;

		const cy = primitiveBounds[ i + 2 ];
		const hy = primitiveBounds[ i + 3 ];
		const ly = cy - hy;
		const ry = cy + hy;
		if ( ly < miny ) miny = ly;
		if ( ry > maxy ) maxy = ry;
		if ( cy < cminy ) cminy = cy;
		if ( cy > cmaxy ) cmaxy = cy;

		const cz = primitiveBounds[ i + 4 ];
		const hz = primitiveBounds[ i + 5 ];
		const lz = cz - hz;
		const rz = cz + hz;
		if ( lz < minz ) minz = lz;
		if ( rz > maxz ) maxz = rz;
		if ( cz < cminz ) cminz = cz;
		if ( cz > cmaxz ) cmaxz = cz;

	}

	target[ 0 ] = minx;
	target[ 1 ] = miny;
	target[ 2 ] = minz;

	target[ 3 ] = maxx;
	target[ 4 ] = maxy;
	target[ 5 ] = maxz;

	centroidTarget[ 0 ] = cminx;
	centroidTarget[ 1 ] = cminy;
	centroidTarget[ 2 ] = cminz;

	centroidTarget[ 3 ] = cmaxx;
	centroidTarget[ 4 ] = cmaxy;
	centroidTarget[ 5 ] = cmaxz;

}

const BIN_COUNT = 32;
const binsSort = ( a, b ) => a.candidate - b.candidate;
const sahBins = /* @__PURE__ */ new Array( BIN_COUNT ).fill().map( () => {

	return {

		count: 0,
		bounds: new Float32Array( 6 ),
		rightCacheBounds: new Float32Array( 6 ),
		leftCacheBounds: new Float32Array( 6 ),
		candidate: 0,

	};

} );
const leftBounds = /* @__PURE__ */ new Float32Array( 6 );

function getOptimalSplit( nodeBoundingData, centroidBoundingData, primitiveBounds, offset, count, strategy ) {

	let axis = - 1;
	let pos = 0;

	// Center
	if ( strategy === CENTER ) {

		axis = getLongestEdgeIndex( centroidBoundingData );
		if ( axis !== - 1 ) {

			pos = ( centroidBoundingData[ axis ] + centroidBoundingData[ axis + 3 ] ) / 2;

		}

	} else if ( strategy === AVERAGE ) {

		axis = getLongestEdgeIndex( nodeBoundingData );
		if ( axis !== - 1 ) {

			pos = getAverage( primitiveBounds, offset, count, axis );

		}

	} else if ( strategy === SAH ) {

		const rootSurfaceArea = computeSurfaceArea( nodeBoundingData );
		let bestCost = PRIMITIVE_INTERSECT_COST * count;

		// iterate over all axes
		const boundsOffset = primitiveBounds.offset || 0;
		const cStart = ( offset - boundsOffset ) * 6;
		const cEnd = ( offset + count - boundsOffset ) * 6;
		for ( let a = 0; a < 3; a ++ ) {

			const axisLeft = centroidBoundingData[ a ];
			const axisRight = centroidBoundingData[ a + 3 ];
			const axisLength = axisRight - axisLeft;
			const binWidth = axisLength / BIN_COUNT;

			// If we have fewer primitives than we're planning to split then just check all
			// the primitive positions because it will be faster.
			if ( count < BIN_COUNT / 4 ) {

				// initialize the bin candidates
				const truncatedBins = [ ...sahBins ];
				truncatedBins.length = count;

				// set the candidates
				let b = 0;
				for ( let c = cStart; c < cEnd; c += 6, b ++ ) {

					const bin = truncatedBins[ b ];
					bin.candidate = primitiveBounds[ c + 2 * a ];
					bin.count = 0;

					const {
						bounds,
						leftCacheBounds,
						rightCacheBounds,
					} = bin;
					for ( let d = 0; d < 3; d ++ ) {

						rightCacheBounds[ d ] = Infinity;
						rightCacheBounds[ d + 3 ] = - Infinity;

						leftCacheBounds[ d ] = Infinity;
						leftCacheBounds[ d + 3 ] = - Infinity;

						bounds[ d ] = Infinity;
						bounds[ d + 3 ] = - Infinity;

					}

					expandByPrimitiveBounds( c, primitiveBounds, bounds );

				}

				truncatedBins.sort( binsSort );

				// remove redundant splits
				let splitCount = count;
				for ( let bi = 0; bi < splitCount; bi ++ ) {

					const bin = truncatedBins[ bi ];
					while ( bi + 1 < splitCount && truncatedBins[ bi + 1 ].candidate === bin.candidate ) {

						truncatedBins.splice( bi + 1, 1 );
						splitCount --;

					}

				}

				// find the appropriate bin for each primitive and expand the bounds.
				for ( let c = cStart; c < cEnd; c += 6 ) {

					const center = primitiveBounds[ c + 2 * a ];
					for ( let bi = 0; bi < splitCount; bi ++ ) {

						const bin = truncatedBins[ bi ];
						if ( center >= bin.candidate ) {

							expandByPrimitiveBounds( c, primitiveBounds, bin.rightCacheBounds );

						} else {

							expandByPrimitiveBounds( c, primitiveBounds, bin.leftCacheBounds );
							bin.count ++;

						}

					}

				}

				// expand all the bounds
				for ( let bi = 0; bi < splitCount; bi ++ ) {

					const bin = truncatedBins[ bi ];
					const leftCount = bin.count;
					const rightCount = count - bin.count;

					// check the cost of this split
					const leftBounds = bin.leftCacheBounds;
					const rightBounds = bin.rightCacheBounds;

					let leftProb = 0;
					if ( leftCount !== 0 ) {

						leftProb = computeSurfaceArea( leftBounds ) / rootSurfaceArea;

					}

					let rightProb = 0;
					if ( rightCount !== 0 ) {

						rightProb = computeSurfaceArea( rightBounds ) / rootSurfaceArea;

					}

					const cost = TRAVERSAL_COST + PRIMITIVE_INTERSECT_COST * (
						leftProb * leftCount + rightProb * rightCount
					);

					if ( cost < bestCost ) {

						axis = a;
						bestCost = cost;
						pos = bin.candidate;

					}

				}

			} else {

				// reset the bins
				for ( let i = 0; i < BIN_COUNT; i ++ ) {

					const bin = sahBins[ i ];
					bin.count = 0;
					bin.candidate = axisLeft + binWidth + i * binWidth;

					const bounds = bin.bounds;
					for ( let d = 0; d < 3; d ++ ) {

						bounds[ d ] = Infinity;
						bounds[ d + 3 ] = - Infinity;

					}

				}

				// iterate over all center positions
				for ( let c = cStart; c < cEnd; c += 6 ) {

					const triCenter = primitiveBounds[ c + 2 * a ];
					const relativeCenter = triCenter - axisLeft;

					// in the partition function if the centroid lies on the split plane then it is
					// considered to be on the right side of the split
					let binIndex = ~ ~ ( relativeCenter / binWidth );
					if ( binIndex >= BIN_COUNT ) binIndex = BIN_COUNT - 1;

					const bin = sahBins[ binIndex ];
					bin.count ++;

					expandByPrimitiveBounds( c, primitiveBounds, bin.bounds );

				}

				// cache the unioned bounds from right to left so we don't have to regenerate them each time
				const lastBin = sahBins[ BIN_COUNT - 1 ];
				copyBounds( lastBin.bounds, lastBin.rightCacheBounds );
				for ( let i = BIN_COUNT - 2; i >= 0; i -- ) {

					const bin = sahBins[ i ];
					const nextBin = sahBins[ i + 1 ];
					unionBounds( bin.bounds, nextBin.rightCacheBounds, bin.rightCacheBounds );

				}

				let leftCount = 0;
				for ( let i = 0; i < BIN_COUNT - 1; i ++ ) {

					const bin = sahBins[ i ];
					const binCount = bin.count;
					const bounds = bin.bounds;

					const nextBin = sahBins[ i + 1 ];
					const rightBounds = nextBin.rightCacheBounds;

					// don't do anything with the bounds if the new bounds have no primitives
					if ( binCount !== 0 ) {

						if ( leftCount === 0 ) {

							copyBounds( bounds, leftBounds );

						} else {

							unionBounds( bounds, leftBounds, leftBounds );

						}

					}

					leftCount += binCount;

					// check the cost of this split
					let leftProb = 0;
					let rightProb = 0;

					if ( leftCount !== 0 ) {

						leftProb = computeSurfaceArea( leftBounds ) / rootSurfaceArea;

					}

					const rightCount = count - leftCount;
					if ( rightCount !== 0 ) {

						rightProb = computeSurfaceArea( rightBounds ) / rootSurfaceArea;

					}

					const cost = TRAVERSAL_COST + PRIMITIVE_INTERSECT_COST * (
						leftProb * leftCount + rightProb * rightCount
					);

					if ( cost < bestCost ) {

						axis = a;
						bestCost = cost;
						pos = bin.candidate;

					}

				}

			}

		}

	} else {

		console.warn( `BVH: Invalid build strategy value ${ strategy } used.` );

	}

	return { axis, pos };

}

// returns the average coordinate on the specified axis of all the provided primitives
function getAverage( primitiveBounds, offset, count, axis ) {

	let avg = 0;
	const boundsOffset = primitiveBounds.offset;
	for ( let i = offset, end = offset + count; i < end; i ++ ) {

		avg += primitiveBounds[ ( i - boundsOffset ) * 6 + axis * 2 ];

	}

	return avg / count;

}

class BVHNode {

	constructor() {

		// internal nodes have boundingData, left, right, and splitAxis
		// leaf nodes have offset and count (referring to primitives in the mesh geometry)

		this.boundingData = new Float32Array( 6 );

	}

}

// reorders the partition buffer such that for `count` elements after `offset`, elements on the left side of the split
// will be on the left and elements on the right side of the split will be on the right. returns the index
// of the first element on the right side, or offset + count if there are no elements on the right side.
function partition( buffer, stride, primitiveBounds, offset, count, split ) {

	let left = offset;
	let right = offset + count - 1;
	const pos = split.pos;
	const axisOffset = split.axis * 2;
	const boundsOffset = primitiveBounds.offset || 0;

	// hoare partitioning, see e.g. https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
	while ( true ) {

		while ( left <= right && primitiveBounds[ ( left - boundsOffset ) * 6 + axisOffset ] < pos ) {

			left ++;

		}

		// if a primitive center lies on the partition plane it is considered to be on the right side
		while ( left <= right && primitiveBounds[ ( right - boundsOffset ) * 6 + axisOffset ] >= pos ) {

			right --;

		}

		if ( left < right ) {

			// we need to swap all of the information associated with the primitives at index
			// left and right; that's the elements in the partition buffer and the bounds
			for ( let i = 0; i < stride; i ++ ) {

				let t0 = buffer[ left * stride + i ];
				buffer[ left * stride + i ] = buffer[ right * stride + i ];
				buffer[ right * stride + i ] = t0;

			}

			// swap bounds
			for ( let i = 0; i < 6; i ++ ) {

				const l = left - boundsOffset;
				const r = right - boundsOffset;
				const tb = primitiveBounds[ l * 6 + i ];
				primitiveBounds[ l * 6 + i ] = primitiveBounds[ r * 6 + i ];
				primitiveBounds[ r * 6 + i ] = tb;

			}

			left ++;
			right --;

		} else {

			return left;

		}

	}

}

let float32Array, uint32Array, uint16Array, uint8Array;
const MAX_POINTER = Math.pow( 2, 32 );

function countNodes( node ) {

	if ( 'count' in node ) {

		return 1;

	} else {

		return 1 + countNodes( node.left ) + countNodes( node.right );

	}

}

function populateBuffer( byteOffset, node, buffer ) {

	float32Array = new Float32Array( buffer );
	uint32Array = new Uint32Array( buffer );
	uint16Array = new Uint16Array( buffer );
	uint8Array = new Uint8Array( buffer );

	return _populateBuffer( byteOffset, node );

}

// pack structure
// boundingData  				: 6 float32
// right / offset 				: 1 uint32
// splitAxis / isLeaf + count 	: 1 uint32 / 2 uint16
function _populateBuffer( byteOffset, node ) {

	const node32Index = byteOffset / 4;
	const node16Index = byteOffset / 2;
	const isLeaf = 'count' in node;
	const boundingData = node.boundingData;
	for ( let i = 0; i < 6; i ++ ) {

		float32Array[ node32Index + i ] = boundingData[ i ];

	}

	if ( isLeaf ) {

		if ( node.buffer ) {

			uint8Array.set( new Uint8Array( node.buffer ), byteOffset );
			return byteOffset + node.buffer.byteLength;

		} else {

			uint32Array[ node32Index + 6 ] = node.offset;
			uint16Array[ node16Index + 14 ] = node.count;
			uint16Array[ node16Index + 15 ] = IS_LEAFNODE_FLAG;
			return byteOffset + BYTES_PER_NODE;

		}

	} else {

		const { left, right, splitAxis } = node;

		// fill in the left node contents
		const leftByteOffset = byteOffset + BYTES_PER_NODE;
		let rightByteOffset = _populateBuffer( leftByteOffset, left );

		// calculate relative offset from parent to right child
		const currentNodeIndex = byteOffset / BYTES_PER_NODE;
		const rightNodeIndex = rightByteOffset / BYTES_PER_NODE;
		const relativeRightIndex = rightNodeIndex - currentNodeIndex;

		// check if the relative offset is too high
		if ( relativeRightIndex > MAX_POINTER ) {

			throw new Error( 'MeshBVH: Cannot store relative child node offset greater than 32 bits.' );

		}

		// fill in the right node contents (store as relative offset)
		uint32Array[ node32Index + 6 ] = relativeRightIndex;
		uint32Array[ node32Index + 7 ] = splitAxis;

		// return the next available buffer pointer
		return _populateBuffer( rightByteOffset, right );

	}

}

function buildTree( bvh, primitiveBounds, offset, count, options, loadRange ) {

	// expand variables
	const {
		maxDepth,
		verbose,
		targetLeafSize,
		_strictLeafSize = Infinity,
		strategy,
		onProgress,
	} = options;

	const partitionBuffer = bvh.primitiveBuffer;
	const partitionStride = bvh.primitiveBufferStride;

	// generate intermediate variables
	const cacheCentroidBoundingData = new Float32Array( 6 );
	let reachedMaxDepth = false;

	const root = new BVHNode();
	getBounds( primitiveBounds, offset, count, root.boundingData, cacheCentroidBoundingData );
	splitNode( root, offset, count, cacheCentroidBoundingData );
	return root;

	function triggerProgress( primitivesProcessed ) {

		if ( onProgress ) {

			onProgress( ( primitivesProcessed - loadRange.offset ) / loadRange.count );

		}

	}

	// either recursively splits the given node, creating left and right subtrees for it, or makes it a leaf node,
	// recording the offset and count of its primitives and writing them into the reordered geometry index.
	function splitNode( node, offset, count, centroidBoundingData = null, depth = 0 ) {

		if ( ! reachedMaxDepth && depth >= maxDepth ) {

			reachedMaxDepth = true;
			if ( verbose ) {

				console.warn( `BVH: Max depth of ${ maxDepth } reached when generating BVH. Consider increasing maxDepth.` );

			}

		}

		// A hard guarantee that no leaf exceeds "_strictLeafSize" primitives. When this node is over
		// that limit it must keep splitting regardless of the heuristic.
		const mustSplit = count > _strictLeafSize;

		// early out if we've met our capacity - unless the strict guarantee still requires a split
		if ( ( count <= targetLeafSize && ! mustSplit ) || depth >= maxDepth ) {

			triggerProgress( offset + count );
			node.offset = offset;
			node.count = count;
			return node;

		}

		// Find where to split the volume
		const split = getOptimalSplit( node.boundingData, centroidBoundingData, primitiveBounds, offset, count, strategy );
		let splitOffset = split.axis === - 1 ? - 1 : partition( partitionBuffer, partitionStride, primitiveBounds, offset, count, split );

		// If the heuristic can't produce a usable split then make a leaf unless the strict guarantee requires the split -
		// in which case force an arbitrary median split. The axis comes from the node bounds so parallel and serial
		// builds produce identical trees.
		if ( split.axis === - 1 || splitOffset === offset || splitOffset === offset + count ) {

			if ( ! mustSplit ) {

				triggerProgress( offset + count );
				node.offset = offset;
				node.count = count;
				return node;

			}

			split.axis = Math.max( 0, getLongestEdgeIndex( node.boundingData ) );
			splitOffset = offset + Math.max( 1, Math.floor( count / 2 ) );

		}

		// create the two new child nodes
		node.splitAxis = split.axis;

		// create the left child and compute its bounding box
		const left = new BVHNode();
		const lstart = offset;
		const lcount = splitOffset - offset;
		node.left = left;

		getBounds( primitiveBounds, lstart, lcount, left.boundingData, cacheCentroidBoundingData );
		splitNode( left, lstart, lcount, cacheCentroidBoundingData, depth + 1 );

		// repeat for right
		const right = new BVHNode();
		const rstart = splitOffset;
		const rcount = count - lcount;
		node.right = right;

		getBounds( primitiveBounds, rstart, rcount, right.boundingData, cacheCentroidBoundingData );
		splitNode( right, rstart, rcount, cacheCentroidBoundingData, depth + 1 );

		return node;

	}

}

function buildPackedTree( bvh, options ) {

	const BufferConstructor = options.useSharedArrayBuffer ? SharedArrayBuffer : ArrayBuffer;

	// get the range of buffer data to construct / arrange
	const rootRanges = bvh.getRootRanges( options.range );
	const firstRange = rootRanges[ 0 ];
	const lastRange = rootRanges[ rootRanges.length - 1 ];
	const fullRange = {
		offset: firstRange.offset,
		count: lastRange.offset + lastRange.count - firstRange.offset,
	};

	// construct the primitive bounds for sorting
	const primitiveBounds = new Float32Array( 6 * fullRange.count );
	primitiveBounds.offset = fullRange.offset;
	bvh.computePrimitiveBounds( fullRange.offset, fullRange.count, primitiveBounds );

	// Build BVH roots
	bvh._roots = rootRanges.map( range => {

		const root = buildTree( bvh, primitiveBounds, range.offset, range.count, options, fullRange );
		const nodeCount = countNodes( root );
		const buffer = new BufferConstructor( BYTES_PER_NODE * nodeCount );
		populateBuffer( 0, root, buffer );
		return buffer;

	} );

}

class PrimitivePool {

	constructor( getNewPrimitive ) {

		this._getNewPrimitive = getNewPrimitive;
		this._primitives = [];

	}

	getPrimitive() {

		const primitives = this._primitives;
		if ( primitives.length === 0 ) {

			return this._getNewPrimitive();

		} else {

			return primitives.pop();

		}

	}

	releasePrimitive( primitive ) {

		this._primitives.push( primitive );

	}

}

class _BufferStack {

	constructor() {

		this.float32Array = null;
		this.uint16Array = null;
		this.uint32Array = null;

		const stack = [];
		let prevBuffer = null;
		this.setBuffer = buffer => {

			if ( prevBuffer ) {

				stack.push( prevBuffer );

			}

			prevBuffer = buffer;
			this.float32Array = new Float32Array( buffer );
			this.uint16Array = new Uint16Array( buffer );
			this.uint32Array = new Uint32Array( buffer );

		};

		this.clearBuffer = () => {

			prevBuffer = null;
			this.float32Array = null;
			this.uint16Array = null;
			this.uint32Array = null;

			if ( stack.length !== 0 ) {

				this.setBuffer( stack.pop() );

			}

		};

	}

}

const BufferStack = /* @__PURE__ */ new _BufferStack();

let _box1$1, _box2$1;
const boxStack = [];
const boxPool = /* @__PURE__ */ new PrimitivePool( () => new Box3() );

function shapecast( bvh, root, intersectsBounds, intersectsRange, boundsTraverseOrder, nodeOffset ) {

	// setup
	_box1$1 = boxPool.getPrimitive();
	_box2$1 = boxPool.getPrimitive();
	boxStack.push( _box1$1, _box2$1 );
	BufferStack.setBuffer( bvh._roots[ root ] );

	const result = shapecastTraverse( 0, bvh.geometry, intersectsBounds, intersectsRange, boundsTraverseOrder, nodeOffset );

	// cleanup
	BufferStack.clearBuffer();
	boxPool.releasePrimitive( _box1$1 );
	boxPool.releasePrimitive( _box2$1 );
	boxStack.pop();
	boxStack.pop();

	const length = boxStack.length;
	if ( length > 0 ) {

		_box2$1 = boxStack[ length - 1 ];
		_box1$1 = boxStack[ length - 2 ];

	}

	return result;

}

function shapecastTraverse(
	nodeIndex32,
	geometry,
	intersectsBoundsFunc,
	intersectsRangeFunc,
	nodeScoreFunc = null,
	nodeIndexOffset = 0, // offset for unique node identifier
	depth = 0
) {

	const { float32Array, uint16Array, uint32Array } = BufferStack;
	let nodeIndex16 = nodeIndex32 * 2;

	const isLeaf = IS_LEAF( nodeIndex16, uint16Array );
	if ( isLeaf ) {

		const offset = OFFSET( nodeIndex32, uint32Array );
		const count = COUNT( nodeIndex16, uint16Array );
		arrayToBox( BOUNDING_DATA_INDEX( nodeIndex32 ), float32Array, _box1$1 );
		return intersectsRangeFunc( offset, count, false, depth, nodeIndexOffset + nodeIndex32 / UINT32_PER_NODE, _box1$1 );

	} else {

		const left = LEFT_NODE( nodeIndex32 );
		const right = RIGHT_NODE( nodeIndex32, uint32Array );
		let c1 = left;
		let c2 = right;

		let score1, score2;
		let box1, box2;
		if ( nodeScoreFunc ) {

			box1 = _box1$1;
			box2 = _box2$1;

			// bounding data is not offset
			arrayToBox( BOUNDING_DATA_INDEX( c1 ), float32Array, box1 );
			arrayToBox( BOUNDING_DATA_INDEX( c2 ), float32Array, box2 );

			score1 = nodeScoreFunc( box1 );
			score2 = nodeScoreFunc( box2 );

			if ( score2 < score1 ) {

				c1 = right;
				c2 = left;

				const temp = score1;
				score1 = score2;
				score2 = temp;

				box1 = box2;
				// box2 is always set before use below

			}

		}

		// Check box 1 intersection
		if ( ! box1 ) {

			box1 = _box1$1;
			arrayToBox( BOUNDING_DATA_INDEX( c1 ), float32Array, box1 );

		}

		const isC1Leaf = IS_LEAF( c1 * 2, uint16Array );
		const c1Intersection = intersectsBoundsFunc( box1, isC1Leaf, score1, depth + 1, nodeIndexOffset + c1 / UINT32_PER_NODE );

		let c1StopTraversal;
		if ( c1Intersection === CONTAINED ) {

			const offset = getLeftOffset( c1 );
			const end = getRightEndOffset( c1 );
			const count = end - offset;

			c1StopTraversal = intersectsRangeFunc( offset, count, true, depth + 1, nodeIndexOffset + c1 / UINT32_PER_NODE, box1 );

		} else {

			c1StopTraversal =
				c1Intersection &&
				shapecastTraverse(
					c1,
					geometry,
					intersectsBoundsFunc,
					intersectsRangeFunc,
					nodeScoreFunc,
					nodeIndexOffset,
					depth + 1
				);

		}

		if ( c1StopTraversal ) return true;

		// Check box 2 intersection
		// cached box2 will have been overwritten by previous traversal
		box2 = _box2$1;
		arrayToBox( BOUNDING_DATA_INDEX( c2 ), float32Array, box2 );

		const isC2Leaf = IS_LEAF( c2 * 2, uint16Array );
		const c2Intersection = intersectsBoundsFunc( box2, isC2Leaf, score2, depth + 1, nodeIndexOffset + c2 / UINT32_PER_NODE );

		let c2StopTraversal;
		if ( c2Intersection === CONTAINED ) {

			const offset = getLeftOffset( c2 );
			const end = getRightEndOffset( c2 );
			const count = end - offset;

			c2StopTraversal = intersectsRangeFunc( offset, count, true, depth + 1, nodeIndexOffset + c2 / UINT32_PER_NODE, box2 );

		} else {

			c2StopTraversal =
				c2Intersection &&
				shapecastTraverse(
					c2,
					geometry,
					intersectsBoundsFunc,
					intersectsRangeFunc,
					nodeScoreFunc,
					nodeIndexOffset,
					depth + 1
				);

		}

		if ( c2StopTraversal ) return true;

		return false;

		// Define these inside the function so it has access to the local variables needed
		// when converting to the buffer equivalents
		function getLeftOffset( nodeIndex32 ) {

			const { uint16Array, uint32Array } = BufferStack;
			let nodeIndex16 = nodeIndex32 * 2;

			// traverse until we find a leaf
			while ( ! IS_LEAF( nodeIndex16, uint16Array ) ) {

				nodeIndex32 = LEFT_NODE( nodeIndex32 );
				nodeIndex16 = nodeIndex32 * 2;

			}

			return OFFSET( nodeIndex32, uint32Array );

		}

		function getRightEndOffset( nodeIndex32 ) {

			const { uint16Array, uint32Array } = BufferStack;
			let nodeIndex16 = nodeIndex32 * 2;

			// traverse until we find a leaf
			while ( ! IS_LEAF( nodeIndex16, uint16Array ) ) {

				// adjust offset to point to the right node
				nodeIndex32 = RIGHT_NODE( nodeIndex32, uint32Array );
				nodeIndex16 = nodeIndex32 * 2;

			}

			// return the end offset of the triangle range
			return OFFSET( nodeIndex32, uint32Array ) + COUNT( nodeIndex16, uint16Array );

		}

	}

}

const _bufferStack1 = /* @__PURE__ */ new BufferStack.constructor();
const _bufferStack2 = /* @__PURE__ */ new BufferStack.constructor();
const _boxPool = /* @__PURE__ */ new PrimitivePool( () => new Box3() );
const _leftBox1 = /* @__PURE__ */ new Box3();
const _rightBox1 = /* @__PURE__ */ new Box3();

const _leftBox2 = /* @__PURE__ */ new Box3();
const _rightBox2 = /* @__PURE__ */ new Box3();

let _active = false;

function bvhcast( bvh, otherBvh, matrixToLocal, intersectsRanges ) {

	if ( _active ) {

		throw new Error( 'MeshBVH: Recursive calls to bvhcast not supported.' );

	}

	_active = true;

	const roots = bvh._roots;
	const otherRoots = otherBvh._roots;
	let result;
	let nodeOffset1 = 0;
	let nodeOffset2 = 0;
	const invMat = new Matrix4().copy( matrixToLocal ).invert();

	// iterate over the first set of roots
	for ( let i = 0, il = roots.length; i < il; i ++ ) {

		_bufferStack1.setBuffer( roots[ i ] );
		nodeOffset2 = 0;

		// prep the initial root box
		const localBox = _boxPool.getPrimitive();
		arrayToBox( BOUNDING_DATA_INDEX( 0 ), _bufferStack1.float32Array, localBox );
		localBox.applyMatrix4( invMat );

		// iterate over the second set of roots
		for ( let j = 0, jl = otherRoots.length; j < jl; j ++ ) {

			_bufferStack2.setBuffer( otherRoots[ j ] );

			result = _traverse(
				0, 0, matrixToLocal, invMat, intersectsRanges,
				nodeOffset1, nodeOffset2, 0, 0,
				localBox,
			);

			_bufferStack2.clearBuffer();
			nodeOffset2 += otherRoots[ j ].byteLength / BYTES_PER_NODE;

			if ( result ) {

				break;

			}

		}

		// release stack info
		_boxPool.releasePrimitive( localBox );
		_bufferStack1.clearBuffer();
		nodeOffset1 += roots[ i ].byteLength / BYTES_PER_NODE;

		if ( result ) {

			break;

		}

	}

	_active = false;
	return result;

}

function _traverse(
	node1Index32,
	node2Index32,
	matrix2to1,
	matrix1to2,
	intersectsRangesFunc,

	// offsets for ids
	node1IndexOffset = 0,
	node2IndexOffset = 0,

	// tree depth
	depth1 = 0,
	depth2 = 0,

	currBox = null,
	reversed = false,

) {

	// get the buffer stacks associated with the current indices
	let bufferStack1, bufferStack2;
	if ( reversed ) {

		bufferStack1 = _bufferStack2;
		bufferStack2 = _bufferStack1;

	} else {

		bufferStack1 = _bufferStack1;
		bufferStack2 = _bufferStack2;

	}

	// get the local instances of the typed buffers
	const
		float32Array1 = bufferStack1.float32Array,
		uint32Array1 = bufferStack1.uint32Array,
		uint16Array1 = bufferStack1.uint16Array,
		float32Array2 = bufferStack2.float32Array,
		uint32Array2 = bufferStack2.uint32Array,
		uint16Array2 = bufferStack2.uint16Array;

	const node1Index16 = node1Index32 * 2;
	const node2Index16 = node2Index32 * 2;
	const isLeaf1 = IS_LEAF( node1Index16, uint16Array1 );
	const isLeaf2 = IS_LEAF( node2Index16, uint16Array2 );
	let result = false;
	if ( isLeaf2 && isLeaf1 ) {

		// if both bounds are leaf nodes then fire the callback if the boxes intersect
		// Note the "nodeIndex" values are just intended to be used as unique identifiers in the tree and
		// not used for accessing data
		if ( reversed ) {

			result = intersectsRangesFunc(
				OFFSET( node2Index32, uint32Array2 ), COUNT( node2Index32 * 2, uint16Array2 ),
				OFFSET( node1Index32, uint32Array1 ), COUNT( node1Index32 * 2, uint16Array1 ),
				depth2, node2IndexOffset + node2Index32 / UINT32_PER_NODE,
				depth1, node1IndexOffset + node1Index32 / UINT32_PER_NODE,
			);

		} else {

			result = intersectsRangesFunc(
				OFFSET( node1Index32, uint32Array1 ), COUNT( node1Index32 * 2, uint16Array1 ),
				OFFSET( node2Index32, uint32Array2 ), COUNT( node2Index32 * 2, uint16Array2 ),
				depth1, node1IndexOffset + node1Index32 / UINT32_PER_NODE,
				depth2, node2IndexOffset + node2Index32 / UINT32_PER_NODE,
			);

		}

	} else if ( isLeaf2 ) {

		// SWAP
		// If we've traversed to the leaf node on the other bvh then we need to swap over
		// to traverse down the first one

		// get the new box to use
		const newBox = _boxPool.getPrimitive();
		arrayToBox( BOUNDING_DATA_INDEX( node2Index32 ), float32Array2, newBox );
		newBox.applyMatrix4( matrix2to1 );

		// get the child bounds to check before traversal
		const cl1 = LEFT_NODE( node1Index32 );
		const cr1 = RIGHT_NODE( node1Index32, uint32Array1 );
		arrayToBox( BOUNDING_DATA_INDEX( cl1 ), float32Array1, _leftBox1 );
		arrayToBox( BOUNDING_DATA_INDEX( cr1 ), float32Array1, _rightBox1 );

		// precompute the intersections otherwise the global boxes will be modified during traversal
		const intersectCl1 = newBox.intersectsBox( _leftBox1 );
		const intersectCr1 = newBox.intersectsBox( _rightBox1 );
		result = (
			intersectCl1 && _traverse(
				node2Index32, cl1, matrix1to2, matrix2to1, intersectsRangesFunc,
				node2IndexOffset, node1IndexOffset, depth2, depth1 + 1,
				newBox, ! reversed,
			)
		) || (
			intersectCr1 && _traverse(
				node2Index32, cr1, matrix1to2, matrix2to1, intersectsRangesFunc,
				node2IndexOffset, node1IndexOffset, depth2, depth1 + 1,
				newBox, ! reversed,
			)
		);

		_boxPool.releasePrimitive( newBox );

	} else {

		// if neither are leaves then we should swap if one of the children does not
		// intersect with the current bounds

		// get the child bounds to check
		const cl2 = LEFT_NODE( node2Index32 );
		const cr2 = RIGHT_NODE( node2Index32, uint32Array2 );
		arrayToBox( BOUNDING_DATA_INDEX( cl2 ), float32Array2, _leftBox2 );
		arrayToBox( BOUNDING_DATA_INDEX( cr2 ), float32Array2, _rightBox2 );

		const leftIntersects = currBox.intersectsBox( _leftBox2 );
		const rightIntersects = currBox.intersectsBox( _rightBox2 );
		if ( leftIntersects && rightIntersects ) {

			// continue to traverse both children if they both intersect
			result = _traverse(
				node1Index32, cl2, matrix2to1, matrix1to2, intersectsRangesFunc,
				node1IndexOffset, node2IndexOffset, depth1, depth2 + 1,
				currBox, reversed,
			) || _traverse(
				node1Index32, cr2, matrix2to1, matrix1to2, intersectsRangesFunc,
				node1IndexOffset, node2IndexOffset, depth1, depth2 + 1,
				currBox, reversed,
			);

		} else if ( leftIntersects ) {

			if ( isLeaf1 ) {

				// if the current box is a leaf then just continue
				result = _traverse(
					node1Index32, cl2, matrix2to1, matrix1to2, intersectsRangesFunc,
					node1IndexOffset, node2IndexOffset, depth1, depth2 + 1,
					currBox, reversed,
				);

			} else {

				// SWAP
				// if only one box intersects then we have to swap to the other bvh to continue
				const newBox = _boxPool.getPrimitive();
				newBox.copy( _leftBox2 ).applyMatrix4( matrix2to1 );

				const cl1 = LEFT_NODE( node1Index32 );
				const cr1 = RIGHT_NODE( node1Index32, uint32Array1 );
				arrayToBox( BOUNDING_DATA_INDEX( cl1 ), float32Array1, _leftBox1 );
				arrayToBox( BOUNDING_DATA_INDEX( cr1 ), float32Array1, _rightBox1 );

				// precompute the intersections otherwise the global boxes will be modified during traversal
				const intersectCl1 = newBox.intersectsBox( _leftBox1 );
				const intersectCr1 = newBox.intersectsBox( _rightBox1 );
				result = (
					intersectCl1 && _traverse(
						cl2, cl1, matrix1to2, matrix2to1, intersectsRangesFunc,
						node2IndexOffset, node1IndexOffset, depth2, depth1 + 1,
						newBox, ! reversed,
					)
				) || (
					intersectCr1 && _traverse(
						cl2, cr1, matrix1to2, matrix2to1, intersectsRangesFunc,
						node2IndexOffset, node1IndexOffset, depth2, depth1 + 1,
						newBox, ! reversed,
					)
				);

				_boxPool.releasePrimitive( newBox );

			}

		} else if ( rightIntersects ) {

			if ( isLeaf1 ) {

				// if the current box is a leaf then just continue
				result = _traverse(
					node1Index32, cr2, matrix2to1, matrix1to2, intersectsRangesFunc,
					node1IndexOffset, node2IndexOffset, depth1, depth2 + 1,
					currBox, reversed,
				);

			} else {

				// SWAP
				// if only one box intersects then we have to swap to the other bvh to continue
				const newBox = _boxPool.getPrimitive();
				newBox.copy( _rightBox2 ).applyMatrix4( matrix2to1 );

				const cl1 = LEFT_NODE( node1Index32 );
				const cr1 = RIGHT_NODE( node1Index32, uint32Array1 );
				arrayToBox( BOUNDING_DATA_INDEX( cl1 ), float32Array1, _leftBox1 );
				arrayToBox( BOUNDING_DATA_INDEX( cr1 ), float32Array1, _rightBox1 );

				// precompute the intersections otherwise the global boxes will be modified during traversal
				const intersectCl1 = newBox.intersectsBox( _leftBox1 );
				const intersectCr1 = newBox.intersectsBox( _rightBox1 );
				result = (
					intersectCl1 && _traverse(
						cr2, cl1, matrix1to2, matrix2to1, intersectsRangesFunc,
						node2IndexOffset, node1IndexOffset, depth2, depth1 + 1,
						newBox, ! reversed,
					)
				) || (
					intersectCr1 && _traverse(
						cr2, cr1, matrix1to2, matrix2to1, intersectsRangesFunc,
						node2IndexOffset, node1IndexOffset, depth2, depth1 + 1,
						newBox, ! reversed,
					)
				);

				_boxPool.releasePrimitive( newBox );

			}

		}

	}

	return result;

}

const BVHTraversalHelper = new ( class {

	constructor() {

		let buffer = null;
		let uint32Array = null;
		let uint16Array = null;
		let traversing = false;

		this.root = null;
		this.buffer = null;
		this.uint32Array = null;
		this.uint16Array = null;

		this.setBVH = ( bvh, root ) => {

			if ( traversing ) {

				throw new Error( 'BVHTraversalHelper: cannot call setBVH during an active traversal.' );

			}

			this.root = root;
			this.buffer = buffer = bvh._roots[ root ];
			this.uint16Array = uint16Array = new Uint16Array( buffer );
			this.uint32Array = uint32Array = new Uint32Array( buffer );

		};

		this.reset = () => {

			this.root = null;
			this.buffer = buffer = null;
			this.uint16Array = uint16Array = null;
			this.uint32Array = uint32Array = null;

		};

		this.getRangeStart = node32Index => {

			let node16Index = node32Index * 2;
			while ( ! IS_LEAF( node16Index, uint16Array ) ) {

				node32Index = LEFT_NODE( node32Index );
				node16Index = node32Index * 2;

			}

			return OFFSET( node32Index, uint32Array );

		};

		this.getRangeEnd = node32Index => {

			let node16Index = node32Index * 2;
			while ( ! IS_LEAF( node16Index, uint16Array ) ) {

				node32Index = RIGHT_NODE( node32Index, uint32Array );
				node16Index = node32Index * 2;

			}

			return OFFSET( node32Index, uint32Array ) + COUNT( node16Index, uint16Array );

		};

		// internal recursive walk - the public "traverseBuffer" wraps this with the re-entrancy guard
		const walk = ( callback, node32Index, depth ) => {

			const node16Index = node32Index * 2;
			const isLeaf = IS_LEAF( node16Index, uint16Array );
			const stopTraversal = callback( depth, isLeaf, node32Index );
			if ( ! stopTraversal && ! isLeaf ) {

				const left = LEFT_NODE( node32Index );
				const right = RIGHT_NODE( node32Index, uint32Array );
				walk( callback, left, depth + 1 );
				walk( callback, right, depth + 1 );

			}

		};

		this.traverseBuffer = callback => {

			if ( traversing ) {

				throw new Error( 'BVHTraversalHelper: cannot start a traversal during an active traversal.' );

			}

			traversing = true;
			try {

				walk( callback, 0, 0 );

			} finally {

				traversing = false;

			}

		};

		this.traverse = callback => {

			this.traverseBuffer( ( depth, isLeaf, node32Index ) => {

				if ( isLeaf ) {

					const node16Index = node32Index * 2;
					const offset = uint32Array[ node32Index + 6 ];
					const count = uint16Array[ node16Index + 14 ];
					return callback( depth, isLeaf, new Float32Array( buffer, node32Index * 4, 6 ), offset, count );

				} else {

					const splitAxis = SPLIT_AXIS( node32Index, uint32Array );
					return callback( depth, isLeaf, new Float32Array( buffer, node32Index * 4, 6 ), splitAxis );

				}

			} );

		};

	}

} )();

/** @import { Matrix4 } from 'three' */

const _tempBox = /* @__PURE__ */ new Box3();
const _tempBuffer = /* @__PURE__ */ new Float32Array( 6 );

/**
 * @callback BoundsTraverseOrderCallback
 * @param {Box3} box
 * @returns {number}
 */

/**
 * @callback IntersectsBoundsCallback
 * @param {Box3} box
 * @param {boolean} isLeaf
 * @param {number|undefined} score
 * @param {number} depth
 * @param {number} nodeIndex
 * @returns {number}
 */

/**
 * @callback IntersectsRangeCallback
 * @param {number} offset
 * @param {number} count
 * @param {boolean} contained
 * @param {number} depth
 * @param {number} nodeIndex
 * @param {Box3} box
 * @returns {boolean}
 */

/**
 * @callback IntersectsRangesCallback
 * @param {number} offset1
 * @param {number} count1
 * @param {number} offset2
 * @param {number} count2
 * @param {number} depth1
 * @param {number} nodeIndex1
 * @param {number} depth2
 * @param {number} nodeIndex2
 * @returns {boolean}
 */

/**
 * Abstract base class for BVH implementations. Provides core tree traversal and spatial query
 * methods. Subclasses implement primitive-specific logic by overriding `writePrimitiveBounds`
 * and related internal methods.
 */
class BVH {

	constructor() {

		this._roots = null;
		this.primitiveBuffer = null;
		this.primitiveBufferStride = null;

	}

	init( options ) {

		options = {
			...DEFAULT_OPTIONS,
			...options,
		};

		if ( 'maxLeafSize' in options ) {

			console.warn( 'BVH: "maxLeafSize" option has been deprecated. Use "targetLeafSize", instead.' );
			options = {
				...options,
				targetLeafSize: options.maxLeafSize,
			};

		}

		buildPackedTree( this, options );

	}

	getRootRanges( /* range */ ) {

		// TODO: can we avoid passing range in here?
		throw new Error( 'BVH: getRootRanges() not implemented' );

	}

	// write the i-th primitive bounds in a 6-value min / max format to the buffer
	// starting at the given "writeOffset"
	writePrimitiveBounds( /* i, buffer, writeOffset */ ) {

		throw new Error( 'BVH: writePrimitiveBounds() not implemented' );

	}

	// writes the union bounds of all primitives in the given range in a min / max format
	// to the buffer
	writePrimitiveRangeBounds( offset, count, targetBuffer, baseIndex ) {

		// Initialize bounds
		let minX = Infinity;
		let minY = Infinity;
		let minZ = Infinity;
		let maxX = - Infinity;
		let maxY = - Infinity;
		let maxZ = - Infinity;

		// compute union of all bounds
		for ( let i = offset, end = offset + count; i < end; i ++ ) {

			this.writePrimitiveBounds( i, _tempBuffer, 0 );

			// compute union
			const [ lx, ly, lz, rx, ry, rz ] = _tempBuffer;
			if ( lx < minX ) minX = lx;
			if ( rx > maxX ) maxX = rx;
			if ( ly < minY ) minY = ly;
			if ( ry > maxY ) maxY = ry;
			if ( lz < minZ ) minZ = lz;
			if ( rz > maxZ ) maxZ = rz;

		}

		// write bounds
		targetBuffer[ baseIndex + 0 ] = minX;
		targetBuffer[ baseIndex + 1 ] = minY;
		targetBuffer[ baseIndex + 2 ] = minZ;
		targetBuffer[ baseIndex + 3 ] = maxX;
		targetBuffer[ baseIndex + 4 ] = maxY;
		targetBuffer[ baseIndex + 5 ] = maxZ;

		return targetBuffer;

	}

	computePrimitiveBounds( offset, count, targetBuffer ) {

		const boundsOffset = targetBuffer.offset || 0;
		for ( let i = offset, end = offset + count; i < end; i ++ ) {

			this.writePrimitiveBounds( i, _tempBuffer, 0 );

			// construction primitive bounds requires a center + half extents format
			const [ lx, ly, lz, rx, ry, rz ] = _tempBuffer;

			const cx = ( lx + rx ) / 2;
			const cy = ( ly + ry ) / 2;
			const cz = ( lz + rz ) / 2;

			const hx = ( rx - lx ) / 2;
			const hy = ( ry - ly ) / 2;
			const hz = ( rz - lz ) / 2;

			const baseIndex = ( i - boundsOffset ) * 6;
			targetBuffer[ baseIndex + 0 ] = cx;
			targetBuffer[ baseIndex + 1 ] = hx + ( Math.abs( cx ) + hx ) * FLOAT32_EPSILON;
			targetBuffer[ baseIndex + 2 ] = cy;
			targetBuffer[ baseIndex + 3 ] = hy + ( Math.abs( cy ) + hy ) * FLOAT32_EPSILON;
			targetBuffer[ baseIndex + 4 ] = cz;
			targetBuffer[ baseIndex + 5 ] = hz + ( Math.abs( cz ) + hz ) * FLOAT32_EPSILON;

		}

		return targetBuffer;

	}

	/**
	 * Adjusts all primitive offsets stored in the BVH leaf nodes by the given value. Useful when
	 * geometry buffers have been shifted or compacted (e.g. when merging geometries).
	 * @param {number} offset
	 */
	shiftPrimitiveOffsets( offset ) {

		const indirectBuffer = this._indirectBuffer;
		if ( indirectBuffer ) {

			// the offsets are embedded in the indirect buffer
			for ( let i = 0, l = indirectBuffer.length; i < l; i ++ ) {

				indirectBuffer[ i ] += offset;

			}

		} else {

			// offsets are embedded in the leaf nodes
			const roots = this._roots;
			for ( let rootIndex = 0; rootIndex < roots.length; rootIndex ++ ) {

				const root = roots[ rootIndex ];
				const uint32Array = new Uint32Array( root );
				const uint16Array = new Uint16Array( root );
				const totalNodes = root.byteLength / BYTES_PER_NODE;
				for ( let node = 0; node < totalNodes; node ++ ) {

					const node32Index = UINT32_PER_NODE * node;
					const node16Index = 2 * node32Index;
					if ( IS_LEAF( node16Index, uint16Array ) ) {

						// offset value
						uint32Array[ node32Index + 6 ] += offset;

					}

				}

			}

		}

	}

	/**
	 * Traverses all nodes of the BVH, invoking a callback for each node.
	 *
	 * For leaf nodes the callback receives `( depth, isLeaf, boundingData, offset, count )`.
	 * For internal nodes it receives `( depth, isLeaf, boundingData, splitAxis )` and may
	 * return `true` to stop descending into that node's children.
	 *
	 * @param {Function} callback
	 * @param {number} [rootIndex=0]
	 */
	traverse( callback, rootIndex = 0 ) {

		BVHTraversalHelper.setBVH( this, rootIndex );
		BVHTraversalHelper.traverse( callback );
		BVHTraversalHelper.reset();

	}

	/**
	 * Refits all BVH node bounds to reflect the current primitive positions. Faster than
	 * rebuilding the BVH but produces a less optimal tree after large vertex deformations.
	 */
	refit( /* nodeIndices = null */ ) {

		// TODO: add support for "nodeIndices"
		// if ( nodeIndices && Array.isArray( nodeIndices ) ) {

		// 	nodeIndices = new Set( nodeIndices );

		// }

		const roots = this._roots;
		for ( let rootIndex = 0, rootCount = roots.length; rootIndex < rootCount; rootIndex ++ ) {

			const buffer = roots[ rootIndex ];
			const uint32Array = new Uint32Array( buffer );
			const uint16Array = new Uint16Array( buffer );
			const float32Array = new Float32Array( buffer );
			const totalNodes = buffer.byteLength / BYTES_PER_NODE;

			// Traverse nodes from right to left so children are updated before parents
			for ( let nodeIndex = totalNodes - 1; nodeIndex >= 0; nodeIndex -- ) {

				const nodeIndex32 = nodeIndex * UINT32_PER_NODE;
				const nodeIndex16 = nodeIndex32 * 2;
				const isLeaf = IS_LEAF( nodeIndex16, uint16Array );

				if ( isLeaf ) {

					// get the bounds
					const offset = OFFSET( nodeIndex32, uint32Array );
					const count = COUNT( nodeIndex16, uint16Array );
					this.writePrimitiveRangeBounds( offset, count, _tempBuffer, 0 );

					// write directly to node bounds (already in min/max format)
					float32Array.set( _tempBuffer, nodeIndex32 );

				} else {

					const left = LEFT_NODE( nodeIndex32 );
					const right = RIGHT_NODE( nodeIndex32, uint32Array );

					// Union the bounds of left and right children
					for ( let i = 0; i < 3; i ++ ) {

						const leftMin = float32Array[ left + i ];
						const leftMax = float32Array[ left + i + 3 ];
						const rightMin = float32Array[ right + i ];
						const rightMax = float32Array[ right + i + 3 ];

						float32Array[ nodeIndex32 + i ] = leftMin < rightMin ? leftMin : rightMin;
						float32Array[ nodeIndex32 + i + 3 ] = leftMax > rightMax ? leftMax : rightMax;

					}

				}

			}

		}

	}

	/**
	 * Computes the axis-aligned bounding box of all primitives in the BVH.
	 * @param {Box3} target - Target box to write the result into.
	 * @returns {Box3}
	 */
	getBoundingBox( target ) {

		target.makeEmpty();

		const roots = this._roots;
		roots.forEach( buffer => {

			arrayToBox( 0, new Float32Array( buffer ), _tempBox );
			target.union( _tempBox );

		} );

		return target;

	}

	/**
	 * A generalized traversal function for performing spatial queries against the BVH. Returns
	 * `true` as soon as a primitive has been reported as intersected. The tree is traversed
	 * depth-first; `boundsTraverseOrder` controls which child is visited first. Returning
	 * `CONTAINED` from `intersectsBounds` skips further child traversal and intersects all
	 * primitives in that subtree immediately.
	 *
	 * @param {Object} callbacks
	 * @param {IntersectsBoundsCallback} callbacks.intersectsBounds
	 * @param {IntersectsRangeCallback} [callbacks.intersectsRange]
	 * @param {BoundsTraverseOrderCallback} [callbacks.boundsTraverseOrder]
	 * @returns {boolean}
	 */
	// TODO: see if we can get rid of "iterateFunc" here as well as the primitive so the function
	// API aligns with the "shapecast" implementation
	shapecast( callbacks ) {

		// TODO: can we get rid of "scratchPrimitive" and / or "iterate"? Or merge them somehow
		let {
			boundsTraverseOrder,
			intersectsBounds,
			intersectsRange,
			intersectsPrimitive,
			scratchPrimitive,
			iterate,
		} = callbacks;

		// wrap the intersectsRange function
		if ( intersectsRange && intersectsPrimitive ) {

			const originalIntersectsRange = intersectsRange;
			intersectsRange = ( offset, count, contained, depth, nodeIndex ) => {

				if ( ! originalIntersectsRange( offset, count, contained, depth, nodeIndex ) ) {

					return iterate( offset, count, this, intersectsPrimitive, contained, depth, scratchPrimitive );

				}

				return true;

			};

		} else if ( ! intersectsRange ) {

			if ( intersectsPrimitive ) {

				intersectsRange = ( offset, count, contained, depth ) => {

					return iterate( offset, count, this, intersectsPrimitive, contained, depth, scratchPrimitive );

				};

			} else {

				intersectsRange = ( offset, count, contained ) => {

					return contained;

				};

			}

		}

		// run shapecast
		let result = false;
		let nodeOffset = 0;
		const roots = this._roots;
		for ( let i = 0, l = roots.length; i < l; i ++ ) {

			const root = roots[ i ];
			result = shapecast( this, i, intersectsBounds, intersectsRange, boundsTraverseOrder, nodeOffset );

			if ( result ) {

				break;

			}

			nodeOffset += root.byteLength / BYTES_PER_NODE;

		}

		return result;

	}

	/**
	 * Simultaneously traverses two BVH structures to find intersecting primitive pairs. Returns
	 * `true` as soon as any intersection is reported. Both trees are traversed depth-first with
	 * alternating descent. `matrixToLocal` transforms `otherBvh` into the local space of this BVH.
	 *
	 * @param {BVH} otherBvh
	 * @param {Matrix4} matrixToLocal
	 * @param {Object} callbacks
	 * @param {IntersectsRangesCallback} callbacks.intersectsRanges
	 * @returns {boolean}
	 */
	bvhcast( otherBvh, matrixToLocal, callbacks ) {

		let { intersectsRanges } = callbacks;
		return bvhcast( this, otherBvh, matrixToLocal, intersectsRanges );


	}

}

function isSharedArrayBufferSupported() {

	return typeof SharedArrayBuffer !== 'undefined';

}

function convertToBufferType( array, BufferConstructor ) {

	if ( array === null ) {

		return array;

	} else if ( array.buffer ) {

		const buffer = array.buffer;
		if ( buffer.constructor === BufferConstructor ) {

			return array;

		}

		const ArrayConstructor = array.constructor;
		const result = new ArrayConstructor( new BufferConstructor( buffer.byteLength ) );
		result.set( array );
		return result;

	} else {

		if ( array.constructor === BufferConstructor ) {

			return array;

		}

		const result = new BufferConstructor( array.byteLength );
		new Uint8Array( result ).set( new Uint8Array( array ) );
		return result;

	}

}

function getVertexCount( geo ) {

	return geo.index ? geo.index.count : geo.attributes.position.count;

}

function getTriCount( geo ) {

	return getVertexCount( geo ) / 3;

}

function getIndexArray( vertexCount, BufferConstructor = ArrayBuffer ) {

	if ( vertexCount > 65535 ) {

		return new Uint32Array( new BufferConstructor( 4 * vertexCount ) );

	} else {

		return new Uint16Array( new BufferConstructor( 2 * vertexCount ) );

	}

}

// ensures that an index is present on the geometry
function ensureIndex( geo, options ) {

	if ( ! geo.index ) {

		const vertexCount = geo.attributes.position.count;
		const BufferConstructor = options.useSharedArrayBuffer ? SharedArrayBuffer : ArrayBuffer;
		const index = getIndexArray( vertexCount, BufferConstructor );
		geo.setIndex( new BufferAttribute( index, 1 ) );

		for ( let i = 0; i < vertexCount; i ++ ) {

			index[ i ] = i;

		}

	}

}

// Computes the set of { offset, count } ranges which need independent BVH roots. Each
// region in the geometry index that belongs to a different set of material groups requires
// a separate BVH root, so that triangles indices belonging to one group never get swapped
// with triangle indices belongs to another group. For example, if the groups were like this:
//
// [-------------------------------------------------------------]
// |__________________|
//   g0 = [0, 20]  |______________________||_____________________|
//                      g1 = [16, 40]           g2 = [41, 60]
//
// we would need four BVH roots: [0, 15], [16, 20], [21, 40], [41, 60].
function getFullPrimitiveRange( geo, range, stride ) {

	const primitiveCount = getVertexCount( geo ) / stride;
	const drawRange = range ? range : geo.drawRange;
	const start = drawRange.start / stride;
	const end = ( drawRange.start + drawRange.count ) / stride;

	const offset = Math.max( 0, start );
	const count = Math.min( primitiveCount, end ) - offset;
	return {
		offset: Math.floor( offset ),
		count: Math.floor( count ),
	};

}

function getPrimitiveGroupRanges( geo, stride ) {

	return geo.groups.map( group => ( {
		offset: group.start / stride,
		count: group.count / stride,
	} ));

}

// Function that extracts a set of mutually exclusive ranges representing the primitives being
// drawn as determined by the geometry groups, draw range, and user specified range
function getRootPrimitiveRanges( geo, range, stride ) {

	const drawRange = getFullPrimitiveRange( geo, range, stride );
	const primitiveRanges = getPrimitiveGroupRanges( geo, stride );
	if ( ! primitiveRanges.length ) {

		return [ drawRange ];

	}

	const ranges = [];
	const drawRangeStart = drawRange.offset;
	const drawRangeEnd = drawRange.offset + drawRange.count;

	// Create events for group boundaries
	const primitiveCount = getVertexCount( geo ) / stride;
	const events = [];
	for ( const group of primitiveRanges ) {

		// Account for cases where group size is set to Infinity
		const { offset, count } = group;
		const groupStart = offset;
		const groupCount = isFinite( count ) ? count : ( primitiveCount - offset );
		const groupEnd = ( offset + groupCount );

		// Only add events if the group intersects with the draw range
		if ( groupStart < drawRangeEnd && groupEnd > drawRangeStart ) {

			events.push( { pos: Math.max( drawRangeStart, groupStart ), isStart: true } );
			events.push( { pos: Math.min( drawRangeEnd, groupEnd ), isStart: false } );

		}

	}

	// Sort events by position, with 'end' events before 'start' events at the same position
	events.sort( ( a, b ) => {

		if ( a.pos !== b.pos ) {

			return a.pos - b.pos;

		} else {

			return a.type === 'end' ? - 1 : 1;

		}

	} );

	// sweep through events and create ranges where activeGroups > 0
	let activeGroups = 0;
	let lastPos = null;
	for ( const event of events ) {

		const newPos = event.pos;
		if ( activeGroups !== 0 && newPos !== lastPos ) {

			ranges.push( {
				offset: lastPos,
				count: newPos - lastPos,
			} );

		}

		activeGroups += event.isStart ? 1 : - 1;
		lastPos = newPos;

	}

	return ranges;

}

/** @import { BufferGeometry } from 'three' */

// construct a new buffer that points to the set of triangles represented by the given ranges
function generateIndirectBuffer( ranges, useSharedArrayBuffer ) {

	const lastRange = ranges[ ranges.length - 1 ];
	const useUint32 = lastRange.offset + lastRange.count > 2 ** 16;

	// use getRootIndexRanges which excludes gaps
	const length = ranges.reduce( ( acc, val ) => acc + val.count, 0 );
	const byteCount = useUint32 ? 4 : 2;
	const buffer = useSharedArrayBuffer ? new SharedArrayBuffer( length * byteCount ) : new ArrayBuffer( length * byteCount );
	const indirectBuffer = useUint32 ? new Uint32Array( buffer ) : new Uint16Array( buffer );

	// construct a compact form of the triangles in these ranges
	let index = 0;
	for ( let r = 0; r < ranges.length; r ++ ) {

		const { offset, count } = ranges[ r ];
		for ( let i = 0; i < count; i ++ ) {

			indirectBuffer[ index + i ] = offset + i;

		}

		index += count;

	}

	return indirectBuffer;

}

/**
 * Abstract base class for geometry-backed BVH implementations. Handles geometry
 * indexing, indirect mode, and bounding box initialization. Subclasses implement
 * primitive-specific bounds computation and raycasting via `writePrimitiveBounds`
 * and `raycastObject3D`.
 *
 * @param {BufferGeometry} geometry
 * @param {Object} [options]
 * @param {number} [options.strategy=CENTER] - Split strategy: `CENTER`, `AVERAGE`, or `SAH`.
 * @param {number} [options.maxDepth=40] - Maximum tree depth. Note that this can cause the target leaf size to not
 * be met if the tree is truncated.
 * @param {number} [options.targetLeafSize=10] - The target number of primitives per leaf node. Note that this is
 * a soft limit and generation strategies like SAH will terminate early if the heuristic determines.
 * @param {boolean} [options.setBoundingBox=true] - Set `geometry.boundingBox` if not already present.
 * @param {boolean} [options.useSharedArrayBuffer=false] - Use `SharedArrayBuffer` for BVH root buffers.
 * @param {boolean} [options.indirect=false] - Build using an indirect buffer, leaving the original index unmodified.
 * @param {boolean} [options.verbose=true] - Log build progress to the console.
 * @param {Function|null} [options.onProgress=null] - Called with a progress value in [0, 1] during build.
 * @param {Object|null} [options.range=null] - Restrict the BVH to a specific geometry group range.
 * @extends BVH
 */
class GeometryBVH extends BVH {

	/**
	 * Whether the BVH was built in indirect mode.
	 * @type {boolean}
	 * @readonly
	 */
	get indirect() {

		return ! ! this._indirectBuffer;

	}

	get primitiveStride() {

		return null;

	}

	get primitiveBufferStride() {

		return this.indirect ? 1 : this.primitiveStride;

	}
	set primitiveBufferStride( v ) {}

	get primitiveBuffer() {

		return this.indirect ? this._indirectBuffer : this.geometry.index.array;

	}
	set primitiveBuffer( v ) {}

	constructor( geometry, options = {} ) {

		if ( ! geometry.isBufferGeometry ) {

			throw new Error( 'BVH: Only BufferGeometries are supported.' );

		} else if ( geometry.index && geometry.index.isInterleavedBufferAttribute ) {

			throw new Error( 'BVH: InterleavedBufferAttribute is not supported for the index attribute.' );

		}

		if ( options.useSharedArrayBuffer && ! isSharedArrayBufferSupported() ) {

			throw new Error( 'BVH: SharedArrayBuffer is not available.' );

		}

		super();

		// retain references to the geometry so we can use them it without having to
		// take a geometry reference in every function.

		/**
		 * The geometry this BVH was built from.
		 * @type {BufferGeometry}
		 * @readonly
		 */
		this.geometry = geometry;
		this.resolvePrimitiveIndex = options.indirect ? i => this._indirectBuffer[ i ] : i => i;
		this.primitiveBuffer = null;
		this.primitiveBufferStride = null;
		this._indirectBuffer = null;

		options = {
			...DEFAULT_OPTIONS,
			...options,
		};

		// build the BVH unless we're deserializing
		if ( ! options[ SKIP_GENERATION ] ) {

			this.init( options );

		}

	}

	init( options ) {

		const { geometry, primitiveStride } = this;

		if ( options.indirect ) {

			// construct an buffer that is indirectly sorts the triangles used for the BVH
			const ranges = getRootPrimitiveRanges( geometry, options.range, primitiveStride );
			const indirectBuffer = generateIndirectBuffer( ranges, options.useSharedArrayBuffer );
			this._indirectBuffer = indirectBuffer;

		} else {

			ensureIndex( geometry, options );

		}

		super.init( options );

		if ( ! geometry.boundingBox && options.setBoundingBox ) {

			geometry.boundingBox = this.getBoundingBox( new Box3() );

		}

	}

	// Abstract methods to be implemented by subclasses
	getRootRanges( range ) {

		// TODO: can we avoid passing options in here
		if ( this.indirect ) {

			return [ { offset: 0, count: this._indirectBuffer.length } ];


		} else {

			return getRootPrimitiveRanges( this.geometry, range, this.primitiveStride );

		}

	}

	raycastObject3D( /* object, raycaster, intersects = [] */ ) {

		throw new Error( 'BVH: raycastObject3D() not implemented' );

	}

}

class SeparatingAxisBounds {

	constructor() {

		this.min = Infinity;
		this.max = - Infinity;

	}

	setFromPointsField( points, field ) {

		let min = Infinity;
		let max = - Infinity;
		for ( let i = 0, l = points.length; i < l; i ++ ) {

			const p = points[ i ];
			const val = p[ field ];
			min = val < min ? val : min;
			max = val > max ? val : max;

		}

		this.min = min;
		this.max = max;

	}

	setFromPoints( axis, points ) {

		let min = Infinity;
		let max = - Infinity;
		for ( let i = 0, l = points.length; i < l; i ++ ) {

			const p = points[ i ];
			const val = axis.dot( p );
			min = val < min ? val : min;
			max = val > max ? val : max;

		}

		this.min = min;
		this.max = max;

	}

	isSeparated( other ) {

		return this.min > other.max || other.min > this.max;

	}

}

SeparatingAxisBounds.prototype.setFromBox = ( function () {

	const p = /* @__PURE__ */ new Vector3();
	return function setFromBox( axis, box ) {

		const boxMin = box.min;
		const boxMax = box.max;
		let min = Infinity;
		let max = - Infinity;
		for ( let x = 0; x <= 1; x ++ ) {

			for ( let y = 0; y <= 1; y ++ ) {

				for ( let z = 0; z <= 1; z ++ ) {

					p.x = boxMin.x * x + boxMax.x * ( 1 - x );
					p.y = boxMin.y * y + boxMax.y * ( 1 - y );
					p.z = boxMin.z * z + boxMax.z * ( 1 - z );

					const val = axis.dot( p );
					min = Math.min( val, min );
					max = Math.max( val, max );

				}

			}

		}

		this.min = min;
		this.max = max;

	};

} )();

const areIntersecting = ( function () {

	const cacheSatBounds = /* @__PURE__ */ new SeparatingAxisBounds();
	return function areIntersecting( shape1, shape2 ) {

		const points1 = shape1.points;
		const satAxes1 = shape1.satAxes;
		const satBounds1 = shape1.satBounds;

		const points2 = shape2.points;
		const satAxes2 = shape2.satAxes;
		const satBounds2 = shape2.satBounds;

		// check axes of the first shape
		for ( let i = 0; i < 3; i ++ ) {

			const sb = satBounds1[ i ];
			const sa = satAxes1[ i ];
			cacheSatBounds.setFromPoints( sa, points2 );
			if ( sb.isSeparated( cacheSatBounds ) ) return false;

		}

		// check axes of the second shape
		for ( let i = 0; i < 3; i ++ ) {

			const sb = satBounds2[ i ];
			const sa = satAxes2[ i ];
			cacheSatBounds.setFromPoints( sa, points1 );
			if ( sb.isSeparated( cacheSatBounds ) ) return false;

		}

	};

} )();

const closestPointLineToLine = ( function () {

	// https://github.com/juj/MathGeoLib/blob/master/src/Geometry/Line.cpp#L56
	const dir1 = /* @__PURE__ */ new Vector3();
	const dir2 = /* @__PURE__ */ new Vector3();
	const v02 = /* @__PURE__ */ new Vector3();
	return function closestPointLineToLine( l1, l2, result ) {

		const v0 = l1.start;
		const v10 = dir1;
		const v2 = l2.start;
		const v32 = dir2;

		v02.subVectors( v0, v2 );
		dir1.subVectors( l1.end, l1.start );
		dir2.subVectors( l2.end, l2.start );

		// float d0232 = v02.Dot(v32);
		const d0232 = v02.dot( v32 );

		// float d3210 = v32.Dot(v10);
		const d3210 = v32.dot( v10 );

		// float d3232 = v32.Dot(v32);
		const d3232 = v32.dot( v32 );

		// float d0210 = v02.Dot(v10);
		const d0210 = v02.dot( v10 );

		// float d1010 = v10.Dot(v10);
		const d1010 = v10.dot( v10 );

		// float denom = d1010*d3232 - d3210*d3210;
		const denom = d1010 * d3232 - d3210 * d3210;

		let d, d2;
		if ( denom !== 0 ) {

			d = ( d0232 * d3210 - d0210 * d3232 ) / denom;

		} else {

			d = 0;

		}

		d2 = ( d0232 + d * d3210 ) / d3232;

		result.x = d;
		result.y = d2;

	};

} )();

const closestPointsSegmentToSegment = ( function () {

	// https://github.com/juj/MathGeoLib/blob/master/src/Geometry/LineSegment.cpp#L187
	const paramResult = /* @__PURE__ */ new Vector2();
	const temp1 = /* @__PURE__ */ new Vector3();
	const temp2 = /* @__PURE__ */ new Vector3();
	return function closestPointsSegmentToSegment( l1, l2, target1, target2 ) {

		closestPointLineToLine( l1, l2, paramResult );

		let d = paramResult.x;
		let d2 = paramResult.y;
		if ( d >= 0 && d <= 1 && d2 >= 0 && d2 <= 1 ) {

			l1.at( d, target1 );
			l2.at( d2, target2 );

			return;

		} else if ( d >= 0 && d <= 1 ) {

			// Only d2 is out of bounds.
			if ( d2 < 0 ) {

				l2.at( 0, target2 );

			} else {

				l2.at( 1, target2 );

			}

			l1.closestPointToPoint( target2, true, target1 );
			return;

		} else if ( d2 >= 0 && d2 <= 1 ) {

			// Only d is out of bounds.
			if ( d < 0 ) {

				l1.at( 0, target1 );

			} else {

				l1.at( 1, target1 );

			}

			l2.closestPointToPoint( target1, true, target2 );
			return;

		} else {

			// Both u and u2 are out of bounds.
			let p;
			if ( d < 0 ) {

				p = l1.start;

			} else {

				p = l1.end;

			}

			let p2;
			if ( d2 < 0 ) {

				p2 = l2.start;

			} else {

				p2 = l2.end;

			}

			const closestPoint = temp1;
			const closestPoint2 = temp2;
			l1.closestPointToPoint( p2, true, temp1 );
			l2.closestPointToPoint( p, true, temp2 );

			if ( closestPoint.distanceToSquared( p2 ) <= closestPoint2.distanceToSquared( p ) ) {

				target1.copy( closestPoint );
				target2.copy( p2 );
				return;

			} else {

				target1.copy( p );
				target2.copy( closestPoint2 );
				return;

			}

		}

	};

} )();


const sphereIntersectTriangle = ( function () {

	// https://stackoverflow.com/questions/34043955/detect-collision-between-sphere-and-triangle-in-three-js
	const closestPointTemp = /* @__PURE__ */ new Vector3();
	const projectedPointTemp = /* @__PURE__ */ new Vector3();
	const planeTemp = /* @__PURE__ */ new Plane();
	const lineTemp = /* @__PURE__ */ new Line3();
	return function sphereIntersectTriangle( sphere, triangle ) {

		const { radius, center } = sphere;
		const { a, b, c } = triangle;

		// phase 1
		lineTemp.start = a;
		lineTemp.end = b;
		const closestPoint1 = lineTemp.closestPointToPoint( center, true, closestPointTemp );
		if ( closestPoint1.distanceTo( center ) <= radius ) return true;

		lineTemp.start = a;
		lineTemp.end = c;
		const closestPoint2 = lineTemp.closestPointToPoint( center, true, closestPointTemp );
		if ( closestPoint2.distanceTo( center ) <= radius ) return true;

		lineTemp.start = b;
		lineTemp.end = c;
		const closestPoint3 = lineTemp.closestPointToPoint( center, true, closestPointTemp );
		if ( closestPoint3.distanceTo( center ) <= radius ) return true;

		// phase 2
		const plane = triangle.getPlane( planeTemp );
		const dp = Math.abs( plane.distanceToPoint( center ) );
		if ( dp <= radius ) {

			const pp = plane.projectPoint( center, projectedPointTemp );
			const cp = triangle.containsPoint( pp );
			if ( cp ) return true;

		}

		return false;

	};

} )();

/** @import { Sphere } from 'three' */

const componentKeys = [ 'x', 'y', 'z' ];
const ZERO_EPSILON = 1e-15;
const ZERO_EPSILON_SQR = ZERO_EPSILON * ZERO_EPSILON;
function isNearZero( value ) {

	return Math.abs( value ) < ZERO_EPSILON;

}

/**
 * An extended version of three.js' Triangle class. A variety of derivative values are cached on
 * the object to accelerate the intersection functions. `.needsUpdate` must be set to true when
 * modifying the triangle parameters.
 * @extends Triangle
 */
class ExtendedTriangle extends Triangle {

	constructor( ...args ) {

		super( ...args );

		this.isExtendedTriangle = true;
		this.satAxes = new Array( 4 ).fill().map( () => new Vector3() );
		this.satBounds = new Array( 4 ).fill().map( () => new SeparatingAxisBounds() );
		this.points = [ this.a, this.b, this.c ];
		this.plane = new Plane();
		this.isDegenerateIntoSegment = false;
		this.isDegenerateIntoPoint = false;
		this.degenerateSegment = new Line3();

		/**
		 * Indicates that the triangle fields have changed so cached variables to accelerate other
		 * function execution can be updated. Must be set to true after modifying the triangle
		 * `a`, `b`, `c` fields.
		 * @type {boolean}
		 */
		this.needsUpdate = true;

	}

	/**
	 * Returns whether the triangle intersects the given sphere.
	 * @param {Sphere} sphere
	 * @returns {boolean}
	 */
	intersectsSphere( sphere ) {

		return sphereIntersectTriangle( sphere, this );

	}

	update() {

		const a = this.a;
		const b = this.b;
		const c = this.c;
		const points = this.points;

		const satAxes = this.satAxes;
		const satBounds = this.satBounds;

		const axis0 = satAxes[ 0 ];
		const sab0 = satBounds[ 0 ];
		this.getNormal( axis0 );
		sab0.setFromPoints( axis0, points );

		const axis1 = satAxes[ 1 ];
		const sab1 = satBounds[ 1 ];
		axis1.subVectors( a, b );
		sab1.setFromPoints( axis1, points );

		const axis2 = satAxes[ 2 ];
		const sab2 = satBounds[ 2 ];
		axis2.subVectors( b, c );
		sab2.setFromPoints( axis2, points );

		const axis3 = satAxes[ 3 ];
		const sab3 = satBounds[ 3 ];
		axis3.subVectors( c, a );
		sab3.setFromPoints( axis3, points );

		const lengthAB = axis1.length();
		const lengthBC = axis2.length();
		const lengthCA = axis3.length();

		this.isDegenerateIntoPoint = false;
		this.isDegenerateIntoSegment = false;

		if ( lengthAB < ZERO_EPSILON ) {

			if ( lengthBC < ZERO_EPSILON || lengthCA < ZERO_EPSILON ) {

				this.isDegenerateIntoPoint = true;

			} else {

				this.isDegenerateIntoSegment = true;
				this.degenerateSegment.start.copy( a );
				this.degenerateSegment.end.copy( c );

			}

		} else if ( lengthBC < ZERO_EPSILON ) {

			if ( lengthCA < ZERO_EPSILON ) {

				this.isDegenerateIntoPoint = true;

			} else {

				this.isDegenerateIntoSegment = true;
				this.degenerateSegment.start.copy( b );
				this.degenerateSegment.end.copy( a );

			}

		} else if ( lengthCA < ZERO_EPSILON ) {

			this.isDegenerateIntoSegment = true;
			this.degenerateSegment.start.copy( c );
			this.degenerateSegment.end.copy( b );

		}

		this.plane.setFromNormalAndCoplanarPoint( axis0, a );

		this.needsUpdate = false;

	}

}

/**
 * Returns the distance to the provided line segment. `target1` and `target2` are set to the
 * closest points on the triangle and segment respectively.
 * @function
 * @param {Line3} segment
 * @param {Vector3} [target1]
 * @param {Vector3} [target2]
 * @returns {number}
 */
ExtendedTriangle.prototype.closestPointToSegment = ( function () {

	const point1 = /* @__PURE__ */ new Vector3();
	const point2 = /* @__PURE__ */ new Vector3();
	const edge = /* @__PURE__ */ new Line3();

	return function distanceToSegment( segment, target1 = null, target2 = null ) {

		const { start, end } = segment;
		const points = this.points;
		let distSq;
		let closestDistanceSq = Infinity;

		// check the triangle edges
		for ( let i = 0; i < 3; i ++ ) {

			const nexti = ( i + 1 ) % 3;
			edge.start.copy( points[ i ] );
			edge.end.copy( points[ nexti ] );

			closestPointsSegmentToSegment( edge, segment, point1, point2 );

			distSq = point1.distanceToSquared( point2 );
			if ( distSq < closestDistanceSq ) {

				closestDistanceSq = distSq;
				if ( target1 ) target1.copy( point1 );
				if ( target2 ) target2.copy( point2 );

			}

		}

		// check end points
		this.closestPointToPoint( start, point1 );
		distSq = start.distanceToSquared( point1 );
		if ( distSq < closestDistanceSq ) {

			closestDistanceSq = distSq;
			if ( target1 ) target1.copy( point1 );
			if ( target2 ) target2.copy( start );

		}

		this.closestPointToPoint( end, point1 );
		distSq = end.distanceToSquared( point1 );
		if ( distSq < closestDistanceSq ) {

			closestDistanceSq = distSq;
			if ( target1 ) target1.copy( point1 );
			if ( target2 ) target2.copy( end );

		}

		return Math.sqrt( closestDistanceSq );

	};

} )();

/**
 * Returns whether the triangles intersect. `target` is set to the line segment representing
 * the intersection.
 * @function
 * @param {Triangle} other
 * @param {Line3} [target]
 * @param {boolean} [suppressLog=false]
 * @returns {boolean}
 */
ExtendedTriangle.prototype.intersectsTriangle = ( function () {

	const saTri2 = /* @__PURE__ */ new ExtendedTriangle();
	const cachedSatBounds = /* @__PURE__ */ new SeparatingAxisBounds();
	const cachedSatBounds2 = /* @__PURE__ */ new SeparatingAxisBounds();
	const tmpVec = /* @__PURE__ */ new Vector3();
	const dir1 = /* @__PURE__ */ new Vector3();
	const dir2 = /* @__PURE__ */ new Vector3();
	const tempDir = /* @__PURE__ */ new Vector3();
	const edge1 = /* @__PURE__ */ new Line3();
	const edge2 = /* @__PURE__ */ new Line3();
	const tempPoint = /* @__PURE__ */ new Vector3();
	const bounds1 = /* @__PURE__ */ new Vector2();
	const bounds2 = /* @__PURE__ */ new Vector2();

	function coplanarIntersectsTriangle( self, other, target, suppressLog ) {

		// Perform separating axis intersection test only for coplanar triangles
		// There should be at least one non-degenerate triangle when calling this
		// Otherwise we won't know the plane normal
		const planeNormal = tmpVec;
		if ( ! self.isDegenerateIntoPoint && ! self.isDegenerateIntoSegment ) {

			planeNormal.copy( self.plane.normal );

		} else {

			planeNormal.copy( other.plane.normal );

		}

		const satBounds1 = self.satBounds;
		const satAxes1 = self.satAxes;
		for ( let i = 1; i < 4; i ++ ) {

			const sb = satBounds1[ i ];
			const sa = satAxes1[ i ];
			cachedSatBounds.setFromPoints( sa, other.points );
			if ( sb.isSeparated( cachedSatBounds ) ) return false;

			tempDir.copy( planeNormal ).cross( sa );
			cachedSatBounds.setFromPoints( tempDir, self.points );
			cachedSatBounds2.setFromPoints( tempDir, other.points );
			if ( cachedSatBounds.isSeparated( cachedSatBounds2 ) ) return false;

		}

		const satBounds2 = other.satBounds;
		const satAxes2 = other.satAxes;
		for ( let i = 1; i < 4; i ++ ) {

			const sb = satBounds2[ i ];
			const sa = satAxes2[ i ];
			cachedSatBounds.setFromPoints( sa, self.points );
			if ( sb.isSeparated( cachedSatBounds ) ) return false;

			tempDir.crossVectors( planeNormal, sa );
			cachedSatBounds.setFromPoints( tempDir, self.points );
			cachedSatBounds2.setFromPoints( tempDir, other.points );
			if ( cachedSatBounds.isSeparated( cachedSatBounds2 ) ) return false;

		}

		if ( target ) {

			// TODO find two points that intersect on the edges and make that the result
			if ( ! suppressLog ) {

				console.warn( 'ExtendedTriangle.intersectsTriangle: Triangles are coplanar which does not support an output edge. Setting edge to 0, 0, 0.' );

			}

			target.start.set( 0, 0, 0 );
			target.end.set( 0, 0, 0 );

		}

		return true;

	}

	function findSingleBounds( a, b, c, aProj, bProj, cProj, aDist, bDist, cDist, bounds, edge ) {

		let t = aDist / ( aDist - bDist );
		bounds.x = aProj + ( bProj - aProj ) * t;
		edge.start.subVectors( b, a ).multiplyScalar( t ).add( a );

		t = aDist / ( aDist - cDist );
		bounds.y = aProj + ( cProj - aProj ) * t;
		edge.end.subVectors( c, a ).multiplyScalar( t ).add( a );

	}

	/**
	 * Calculates intersection segment of a triangle with intersection line.
	 * Intersection line is snapped to its biggest component.
	 * And triangle points are passed as a projection on that component.
	 * @returns {boolean} whether this is a coplanar case or not
	 */
	function findIntersectionLineBounds( self, aProj, bProj, cProj, abDist, acDist, aDist, bDist, cDist, bounds, edge ) {

		if ( abDist > 0 ) {

			// then bcDist < 0
			findSingleBounds( self.c, self.a, self.b, cProj, aProj, bProj, cDist, aDist, bDist, bounds, edge );

		} else if ( acDist > 0 ) {

			findSingleBounds( self.b, self.a, self.c, bProj, aProj, cProj, bDist, aDist, cDist, bounds, edge );

		} else if ( bDist * cDist > 0 || aDist != 0 ) {

			findSingleBounds( self.a, self.b, self.c, aProj, bProj, cProj, aDist, bDist, cDist, bounds, edge );

		} else if ( bDist != 0 ) {

			findSingleBounds( self.b, self.a, self.c, bProj, aProj, cProj, bDist, aDist, cDist, bounds, edge );

		} else if ( cDist != 0 ) {

			findSingleBounds( self.c, self.a, self.b, cProj, aProj, bProj, cDist, aDist, bDist, bounds, edge );

		} else {

			return true;

		}

		return false;

	}

	function intersectTriangleSegment( triangle, degenerateTriangle, target, suppressLog ) {

		const segment = degenerateTriangle.degenerateSegment;
		const startDist = triangle.plane.distanceToPoint( segment.start );
		const endDist = triangle.plane.distanceToPoint( segment.end );
		if ( isNearZero( startDist ) ) {

			if ( isNearZero( endDist ) ) {

				return coplanarIntersectsTriangle( triangle, degenerateTriangle, target, suppressLog );

			} else {

				// Is this fine to modify target even if there might be no intersection?
				if ( target ) {

					target.start.copy( segment.start );
					target.end.copy( segment.start );

				}

				return triangle.containsPoint( segment.start );

			}

		} else if ( isNearZero( endDist ) ) {

			if ( target ) {

				target.start.copy( segment.end );
				target.end.copy( segment.end );

			}

			return triangle.containsPoint( segment.end );

		} else {

			if ( triangle.plane.intersectLine( segment, tmpVec ) != null ) {

				if ( target ) {

					target.start.copy( tmpVec );
					target.end.copy( tmpVec );

				}

				return triangle.containsPoint( tmpVec );

			} else {

				return false;

			}

		}

	}

	function intersectTrianglePoint( triangle, degenerateTriangle, target ) {

		const point = degenerateTriangle.a;

		if ( isNearZero( triangle.plane.distanceToPoint( point ) ) && triangle.containsPoint( point ) ) {

			if ( target ) {

				target.start.copy( point );
				target.end.copy( point );

			}

			return true;

		} else {

			return false;

		}

	}

	function intersectSegmentPoint( segmentTri, pointTri, target ) {

		const segment = segmentTri.degenerateSegment;
		const point = pointTri.a;

		segment.closestPointToPoint( point, true, tmpVec );

		if ( point.distanceToSquared( tmpVec ) < ZERO_EPSILON_SQR ) {

			if ( target ) {

				target.start.copy( point );
				target.end.copy( point );

			}

			return true;

		} else {

			return false;

		}

	}

	function handleDegenerateCases( self, other, target, suppressLog ) {

		if ( self.isDegenerateIntoSegment ) {

			if ( other.isDegenerateIntoSegment ) {

				// TODO: replace with Line.distanceSqToLine3 after r179
				const segment1 = self.degenerateSegment;
				const segment2 = other.degenerateSegment;
				const delta1 = dir1;
				const delta2 = dir2;
				segment1.delta( delta1 );
				segment2.delta( delta2 );
				const startDelta = tmpVec.subVectors( segment2.start, segment1.start );

				const denom = delta1.x * delta2.y - delta1.y * delta2.x;
				if ( isNearZero( denom ) ) {

					return false;

				}

				const t = ( startDelta.x * delta2.y - startDelta.y * delta2.x ) / denom;
				const u = - ( delta1.x * startDelta.y - delta1.y * startDelta.x ) / denom;

				if ( t < 0 || t > 1 || u < 0 || u > 1 ) {

					return false;

				}

				const z1 = segment1.start.z + delta1.z * t;
				const z2 = segment2.start.z + delta2.z * u;

				if ( isNearZero( z1 - z2 ) ) {

					if ( target ) {

						target.start.copy( segment1.start ).addScaledVector( delta1, t );
						target.end.copy( segment1.start ).addScaledVector( delta1, t );

					}

					return true;

				} else {

					return false;

				}

			} else if ( other.isDegenerateIntoPoint ) {

				return intersectSegmentPoint( self, other, target );

			} else {

				return intersectTriangleSegment( other, self, target, suppressLog );

			}

		} else if ( self.isDegenerateIntoPoint ) {

			if ( other.isDegenerateIntoPoint ) {

				if ( other.a.distanceToSquared( self.a ) < ZERO_EPSILON_SQR ) {

					if ( target ) {

						target.start.copy( self.a );
						target.end.copy( self.a );

					}

					return true;

				} else {

					return false;

				}

			} else if ( other.isDegenerateIntoSegment ) {

				return intersectSegmentPoint( other, self, target );

			} else {

				return intersectTrianglePoint( other, self, target );

			}

		} else {

			if ( other.isDegenerateIntoPoint ) {

				return intersectTrianglePoint( self, other, target );

			} else if ( other.isDegenerateIntoSegment ) {

				return intersectTriangleSegment( self, other, target, suppressLog );

			} /* else this is a general triangle-traingle case, so return undefined */

		}

	}

	/* TODO: If the triangles are coplanar and intersecting the target is nonsensical. It should at least
	 * be a line contained by both triangles if not a different special case somehow represented in the return result.
	 *
	 * General triangle intersection code is based on Moller's algorithm from here: https://web.stanford.edu/class/cs277/resources/papers/Moller1997b.pdf
	 * Reference implementation from here: https://github.com/erich666/jgt-code/blob/master/Volume_08/Number_1/Shen2003/tri_tri_test/include/Moller97.c#L570
	 * All degeneracies are handled before the general algorithm.
	 * Coplanar check is different from Moller's and based on SAT tests.
	 */
	return function intersectsTriangle( other, target = null, suppressLog = false ) {

		if ( this.needsUpdate ) {

			this.update();

		}

		if ( ! other.isExtendedTriangle ) {

			saTri2.copy( other );
			saTri2.update();
			other = saTri2;

		} else if ( other.needsUpdate ) {

			other.update();

		}

		const res = handleDegenerateCases( this, other, target, suppressLog );
		if ( res !== undefined ) {

			return res;

		}

		const plane1 = this.plane;
		const plane2 = other.plane;

		let a1Dist = plane2.distanceToPoint( this.a );
		let b1Dist = plane2.distanceToPoint( this.b );
		let c1Dist = plane2.distanceToPoint( this.c );

		if ( isNearZero( a1Dist ) )
			a1Dist = 0;

		if ( isNearZero( b1Dist ) )
			b1Dist = 0;

		if ( isNearZero( c1Dist ) )
			c1Dist = 0;

		const a1b1Dist = a1Dist * b1Dist;
		const a1c1Dist = a1Dist * c1Dist;
		if ( a1b1Dist > 0 && a1c1Dist > 0 ) {

			return false;

		}

		let a2Dist = plane1.distanceToPoint( other.a );
		let b2Dist = plane1.distanceToPoint( other.b );
		let c2Dist = plane1.distanceToPoint( other.c );

		if ( isNearZero( a2Dist ) )
			a2Dist = 0;

		if ( isNearZero( b2Dist ) )
			b2Dist = 0;

		if ( isNearZero( c2Dist ) )
			c2Dist = 0;

		const a2b2Dist = a2Dist * b2Dist;
		const a2c2Dist = a2Dist * c2Dist;
		if ( a2b2Dist > 0 && a2c2Dist > 0 ) {

			return false;

		}

		dir1.copy( plane1.normal );
		dir2.copy( plane2.normal );
		const intersectionLine = dir1.cross( dir2 );

		let componentIndex = 0;
		let maxComponent = Math.abs( intersectionLine.x );
		const comp1 = Math.abs( intersectionLine.y );
		if ( comp1 > maxComponent ) {

			maxComponent = comp1;
			componentIndex = 1;

		}

		const comp2 = Math.abs( intersectionLine.z );
		if ( comp2 > maxComponent ) {

			componentIndex = 2;

		}

		const key = componentKeys[ componentIndex ];
		const a1Proj = this.a[ key ];
		const b1Proj = this.b[ key ];
		const c1Proj = this.c[ key ];

		const a2Proj = other.a[ key ];
		const b2Proj = other.b[ key ];
		const c2Proj = other.c[ key ];

		if ( findIntersectionLineBounds( this, a1Proj, b1Proj, c1Proj, a1b1Dist, a1c1Dist, a1Dist, b1Dist, c1Dist, bounds1, edge1 ) ) {

			return coplanarIntersectsTriangle( this, other, target, suppressLog );

		}

		if ( findIntersectionLineBounds( other, a2Proj, b2Proj, c2Proj, a2b2Dist, a2c2Dist, a2Dist, b2Dist, c2Dist, bounds2, edge2 ) ) {

			return coplanarIntersectsTriangle( this, other, target, suppressLog );

		}

		if ( bounds1.y < bounds1.x ) {

			const tmp = bounds1.y;
			bounds1.y = bounds1.x;
			bounds1.x = tmp;

			tempPoint.copy( edge1.start );
			edge1.start.copy( edge1.end );
			edge1.end.copy( tempPoint );

		}

		if ( bounds2.y < bounds2.x ) {

			const tmp = bounds2.y;
			bounds2.y = bounds2.x;
			bounds2.x = tmp;

			tempPoint.copy( edge2.start );
			edge2.start.copy( edge2.end );
			edge2.end.copy( tempPoint );

		}

		if ( bounds1.y < bounds2.x || bounds2.y < bounds1.x ) {

			return false;

		}

		if ( target ) {

			if ( bounds2.x > bounds1.x ) {

				target.start.copy( edge2.start );

			} else {

				target.start.copy( edge1.start );

			}

			if ( bounds2.y < bounds1.y ) {

				target.end.copy( edge2.end );

			} else {

				target.end.copy( edge1.end );

			}

		}

		return true;

	};

} )();


/**
 * Returns the distance to the provided point.
 * @function
 * @param {Vector3} point
 * @returns {number}
 */
ExtendedTriangle.prototype.distanceToPoint = ( function () {

	const target = /* @__PURE__ */ new Vector3();
	return function distanceToPoint( point ) {

		this.closestPointToPoint( point, target );
		return point.distanceTo( target );

	};

} )();


/**
 * Returns the distance to the provided triangle.
 * @function
 * @param {Triangle} other
 * @param {Vector3} [target1]
 * @param {Vector3} [target2]
 * @returns {number}
 */
ExtendedTriangle.prototype.distanceToTriangle = ( function () {

	const point = /* @__PURE__ */ new Vector3();
	const point2 = /* @__PURE__ */ new Vector3();
	const cornerFields = [ 'a', 'b', 'c' ];
	const line1 = /* @__PURE__ */ new Line3();
	const line2 = /* @__PURE__ */ new Line3();

	return function distanceToTriangle( other, target1 = null, target2 = null ) {

		const lineTarget = target1 || target2 ? line1 : null;
		// `intersectsTriangle` returns a zero-length segment for coplanar
		// intersecting triangles, which is fine for our purposes here. We
		// don't need the segment itself in this function. So we can suppress
		// the warning about coplanar triangles.
		if ( this.intersectsTriangle( other, lineTarget, true ) ) {

			if ( target1 || target2 ) {

				if ( target1 ) lineTarget.getCenter( target1 );
				if ( target2 ) lineTarget.getCenter( target2 );

			}

			return 0;

		}

		let closestDistanceSq = Infinity;

		// check all point distances
		for ( let i = 0; i < 3; i ++ ) {

			let dist;
			const field = cornerFields[ i ];
			const otherVec = other[ field ];
			this.closestPointToPoint( otherVec, point );

			dist = otherVec.distanceToSquared( point );

			if ( dist < closestDistanceSq ) {

				closestDistanceSq = dist;
				if ( target1 ) target1.copy( point );
				if ( target2 ) target2.copy( otherVec );

			}


			const thisVec = this[ field ];
			other.closestPointToPoint( thisVec, point );

			dist = thisVec.distanceToSquared( point );

			if ( dist < closestDistanceSq ) {

				closestDistanceSq = dist;
				if ( target1 ) target1.copy( thisVec );
				if ( target2 ) target2.copy( point );

			}

		}

		for ( let i = 0; i < 3; i ++ ) {

			const f11 = cornerFields[ i ];
			const f12 = cornerFields[ ( i + 1 ) % 3 ];
			line1.set( this[ f11 ], this[ f12 ] );
			for ( let i2 = 0; i2 < 3; i2 ++ ) {

				const f21 = cornerFields[ i2 ];
				const f22 = cornerFields[ ( i2 + 1 ) % 3 ];
				line2.set( other[ f21 ], other[ f22 ] );

				closestPointsSegmentToSegment( line1, line2, point, point2 );

				const dist = point.distanceToSquared( point2 );
				if ( dist < closestDistanceSq ) {

					closestDistanceSq = dist;
					if ( target1 ) target1.copy( point );
					if ( target2 ) target2.copy( point2 );

				}

			}

		}

		return Math.sqrt( closestDistanceSq );

	};

} )();

/** @import { Box3, Triangle } from 'three' */

/**
 * An oriented version of three.js' Box3 class. A variety of derivative values are cached on the
 * object to accelerate the intersection functions. `.needsUpdate` must be set to true when
 * modifying the box parameters.
 *
 * @param {Vector3} [min]
 * @param {Vector3} [max]
 * @param {Matrix4} [matrix]
 */
class OrientedBox {

	constructor( min, max, matrix ) {

		this.isOrientedBox = true;

		/** @type {Vector3} */
		this.min = new Vector3();

		/** @type {Vector3} */
		this.max = new Vector3();

		/**
		 * Matrix transformation applied to the box.
		 * @type {Matrix4}
		 */
		this.matrix = new Matrix4();
		this.invMatrix = new Matrix4();
		this.points = new Array( 8 ).fill().map( () => new Vector3() );
		this.satAxes = new Array( 3 ).fill().map( () => new Vector3() );
		this.satBounds = new Array( 3 ).fill().map( () => new SeparatingAxisBounds() );
		this.alignedSatBounds = new Array( 3 ).fill().map( () => new SeparatingAxisBounds() );
		/**
		 * Indicates that the bounding box fields have changed so cached variables to accelerate
		 * other function execution can be updated. Must be set to true after modifying the
		 * oriented box `min`, `max`, `matrix` fields.
		 * @type {boolean}
		 */
		this.needsUpdate = false;

		if ( min ) this.min.copy( min );
		if ( max ) this.max.copy( max );
		if ( matrix ) this.matrix.copy( matrix );

	}

	/**
	 * Sets the oriented box parameters.
	 * @param {Vector3} min
	 * @param {Vector3} max
	 * @param {Matrix4} matrix
	 */
	set( min, max, matrix ) {

		this.min.copy( min );
		this.max.copy( max );
		this.matrix.copy( matrix );
		this.needsUpdate = true;

	}

	copy( other ) {

		this.min.copy( other.min );
		this.max.copy( other.max );
		this.matrix.copy( other.matrix );
		this.needsUpdate = true;

	}

}

OrientedBox.prototype.update = ( function () {

	return function update() {

		const matrix = this.matrix;
		const min = this.min;
		const max = this.max;

		const points = this.points;
		for ( let x = 0; x <= 1; x ++ ) {

			for ( let y = 0; y <= 1; y ++ ) {

				for ( let z = 0; z <= 1; z ++ ) {

					const i = ( ( 1 << 0 ) * x ) | ( ( 1 << 1 ) * y ) | ( ( 1 << 2 ) * z );
					const v = points[ i ];
					v.x = x ? max.x : min.x;
					v.y = y ? max.y : min.y;
					v.z = z ? max.z : min.z;

					v.applyMatrix4( matrix );

				}

			}

		}

		const satBounds = this.satBounds;
		const satAxes = this.satAxes;
		const minVec = points[ 0 ];
		for ( let i = 0; i < 3; i ++ ) {

			const axis = satAxes[ i ];
			const sb = satBounds[ i ];
			const index = 1 << i;
			const pi = points[ index ];

			axis.subVectors( minVec, pi );
			sb.setFromPoints( axis, points );

		}

		const alignedSatBounds = this.alignedSatBounds;
		alignedSatBounds[ 0 ].setFromPointsField( points, 'x' );
		alignedSatBounds[ 1 ].setFromPointsField( points, 'y' );
		alignedSatBounds[ 2 ].setFromPointsField( points, 'z' );

		this.invMatrix.copy( this.matrix ).invert();
		this.needsUpdate = false;

	};

} )();

/**
 * Returns true if intersecting with the provided box.
 * @function
 * @param {Box3} box
 * @returns {boolean}
 */
OrientedBox.prototype.intersectsBox = ( function () {

	const aabbBounds = /* @__PURE__ */ new SeparatingAxisBounds();
	return function intersectsBox( box ) {

		// TODO: should this be doing SAT against the AABB?
		if ( this.needsUpdate ) {

			this.update();

		}

		const min = box.min;
		const max = box.max;
		const satBounds = this.satBounds;
		const satAxes = this.satAxes;
		const alignedSatBounds = this.alignedSatBounds;

		aabbBounds.min = min.x;
		aabbBounds.max = max.x;
		if ( alignedSatBounds[ 0 ].isSeparated( aabbBounds ) ) return false;

		aabbBounds.min = min.y;
		aabbBounds.max = max.y;
		if ( alignedSatBounds[ 1 ].isSeparated( aabbBounds ) ) return false;

		aabbBounds.min = min.z;
		aabbBounds.max = max.z;
		if ( alignedSatBounds[ 2 ].isSeparated( aabbBounds ) ) return false;

		for ( let i = 0; i < 3; i ++ ) {

			const axis = satAxes[ i ];
			const sb = satBounds[ i ];
			aabbBounds.setFromBox( axis, box );
			if ( sb.isSeparated( aabbBounds ) ) return false;

		}

		return true;

	};

} )();

/**
 * Returns true if intersecting with the provided triangle.
 * @function
 * @param {Triangle} triangle
 * @returns {boolean}
 */
OrientedBox.prototype.intersectsTriangle = ( function () {

	const saTri = /* @__PURE__ */ new ExtendedTriangle();
	const pointsArr = /* @__PURE__ */ new Array( 3 );
	const cachedSatBounds = /* @__PURE__ */ new SeparatingAxisBounds();
	const cachedSatBounds2 = /* @__PURE__ */ new SeparatingAxisBounds();
	const cachedAxis = /* @__PURE__ */ new Vector3();
	return function intersectsTriangle( triangle ) {

		if ( this.needsUpdate ) {

			this.update();

		}

		if ( ! triangle.isExtendedTriangle ) {

			saTri.copy( triangle );
			saTri.update();
			triangle = saTri;

		} else if ( triangle.needsUpdate ) {

			triangle.update();

		}

		const satBounds = this.satBounds;
		const satAxes = this.satAxes;

		pointsArr[ 0 ] = triangle.a;
		pointsArr[ 1 ] = triangle.b;
		pointsArr[ 2 ] = triangle.c;

		for ( let i = 0; i < 3; i ++ ) {

			const sb = satBounds[ i ];
			const sa = satAxes[ i ];
			cachedSatBounds.setFromPoints( sa, pointsArr );
			if ( sb.isSeparated( cachedSatBounds ) ) return false;

		}

		const triSatBounds = triangle.satBounds;
		const triSatAxes = triangle.satAxes;
		const points = this.points;
		for ( let i = 0; i < 3; i ++ ) {

			const sb = triSatBounds[ i ];
			const sa = triSatAxes[ i ];
			cachedSatBounds.setFromPoints( sa, points );
			if ( sb.isSeparated( cachedSatBounds ) ) return false;

		}

		// check crossed axes
		for ( let i = 0; i < 3; i ++ ) {

			const sa1 = satAxes[ i ];
			for ( let i2 = 0; i2 < 4; i2 ++ ) {

				const sa2 = triSatAxes[ i2 ];
				cachedAxis.crossVectors( sa1, sa2 );
				cachedSatBounds.setFromPoints( cachedAxis, pointsArr );
				cachedSatBounds2.setFromPoints( cachedAxis, points );
				if ( cachedSatBounds.isSeparated( cachedSatBounds2 ) ) return false;

			}

		}

		return true;

	};

} )();

/**
 * Returns the distance to the provided point. Sets `target` to the closest point on the surface
 * of the box if provided.
 * @function
 * @param {Vector3} point
 * @param {Vector3} target
 * @returns {number}
 */
OrientedBox.prototype.closestPointToPoint = ( function () {

	return function closestPointToPoint( point, target1 ) {

		if ( this.needsUpdate ) {

			this.update();

		}

		target1
			.copy( point )
			.applyMatrix4( this.invMatrix )
			.clamp( this.min, this.max )
			.applyMatrix4( this.matrix );

		return target1;

	};

} )();

/**
 * Returns the distance to the provided point.
 * @function
 * @param {Vector3} point
 * @returns {number}
 */
OrientedBox.prototype.distanceToPoint = ( function () {

	const target = new Vector3();
	return function distanceToPoint( point ) {

		this.closestPointToPoint( point, target );
		return point.distanceTo( target );

	};

} )();

/**
 * Returns the distance to the provided box. `threshold` is an optional distance to return early
 * if the distance is found to be within it. `target1` and `target2` are set to the points on the
 * surface of this box and the `box` argument respectively.
 * @function
 * @param {Box3} box
 * @param {number} [threshold=0]
 * @param {Vector3} [target1]
 * @param {Vector3} [target2]
 * @returns {number}
 */
OrientedBox.prototype.distanceToBox = ( function () {

	const xyzFields = [ 'x', 'y', 'z' ];
	const segments1 = /* @__PURE__ */ new Array( 12 ).fill().map( () => new Line3() );
	const segments2 = /* @__PURE__ */ new Array( 12 ).fill().map( () => new Line3() );

	const point1 = /* @__PURE__ */ new Vector3();
	const point2 = /* @__PURE__ */ new Vector3();

	// early out if we find a value below threshold
	return function distanceToBox( box, threshold = 0, target1 = null, target2 = null ) {

		if ( this.needsUpdate ) {

			this.update();

		}

		if ( this.intersectsBox( box ) ) {

			if ( target1 || target2 ) {

				box.getCenter( point2 );
				this.closestPointToPoint( point2, point1 );
				box.closestPointToPoint( point1, point2 );

				if ( target1 ) target1.copy( point1 );
				if ( target2 ) target2.copy( point2 );

			}

			return 0;

		}

		const threshold2 = threshold * threshold;
		const min = box.min;
		const max = box.max;
		const points = this.points;


		// iterate over every edge and compare distances
		let closestDistanceSq = Infinity;

		// check over all these points
		for ( let i = 0; i < 8; i ++ ) {

			const p = points[ i ];
			point2.copy( p ).clamp( min, max );

			const dist = p.distanceToSquared( point2 );
			if ( dist < closestDistanceSq ) {

				closestDistanceSq = dist;
				if ( target1 ) target1.copy( p );
				if ( target2 ) target2.copy( point2 );

				if ( dist < threshold2 ) return Math.sqrt( dist );

			}

		}

		// generate and check all line segment distances
		let count = 0;
		for ( let i = 0; i < 3; i ++ ) {

			for ( let i1 = 0; i1 <= 1; i1 ++ ) {

				for ( let i2 = 0; i2 <= 1; i2 ++ ) {

					const nextIndex = ( i + 1 ) % 3;
					const nextIndex2 = ( i + 2 ) % 3;

					// get obb line segments
					const index = i1 << nextIndex | i2 << nextIndex2;
					const index2 = 1 << i | i1 << nextIndex | i2 << nextIndex2;
					const p1 = points[ index ];
					const p2 = points[ index2 ];
					const line1 = segments1[ count ];
					line1.set( p1, p2 );


					// get aabb line segments
					const f1 = xyzFields[ i ];
					const f2 = xyzFields[ nextIndex ];
					const f3 = xyzFields[ nextIndex2 ];
					const line2 = segments2[ count ];
					const start = line2.start;
					const end = line2.end;

					start[ f1 ] = min[ f1 ];
					start[ f2 ] = i1 ? min[ f2 ] : max[ f2 ];
					start[ f3 ] = i2 ? min[ f3 ] : max[ f2 ];

					end[ f1 ] = max[ f1 ];
					end[ f2 ] = i1 ? min[ f2 ] : max[ f2 ];
					end[ f3 ] = i2 ? min[ f3 ] : max[ f2 ];

					count ++;

				}

			}

		}

		// check all the other boxes point
		for ( let x = 0; x <= 1; x ++ ) {

			for ( let y = 0; y <= 1; y ++ ) {

				for ( let z = 0; z <= 1; z ++ ) {

					point2.x = x ? max.x : min.x;
					point2.y = y ? max.y : min.y;
					point2.z = z ? max.z : min.z;

					this.closestPointToPoint( point2, point1 );
					const dist = point2.distanceToSquared( point1 );
					if ( dist < closestDistanceSq ) {

						closestDistanceSq = dist;
						if ( target1 ) target1.copy( point1 );
						if ( target2 ) target2.copy( point2 );

						if ( dist < threshold2 ) return Math.sqrt( dist );

					}

				}

			}

		}

		for ( let i = 0; i < 12; i ++ ) {

			const l1 = segments1[ i ];
			for ( let i2 = 0; i2 < 12; i2 ++ ) {

				const l2 = segments2[ i2 ];
				closestPointsSegmentToSegment( l1, l2, point1, point2 );
				const dist = point1.distanceToSquared( point2 );
				if ( dist < closestDistanceSq ) {

					closestDistanceSq = dist;
					if ( target1 ) target1.copy( point1 );
					if ( target2 ) target2.copy( point2 );

					if ( dist < threshold2 ) return Math.sqrt( dist );

				}

			}

		}

		return Math.sqrt( closestDistanceSq );

	};

} )();

class ExtendedTrianglePoolBase extends PrimitivePool {

	constructor() {

		super( () => new ExtendedTriangle() );

	}

}

const ExtendedTrianglePool = /* @__PURE__ */ new ExtendedTrianglePoolBase();

const temp = /* @__PURE__ */ new Vector3();
const temp1$2 = /* @__PURE__ */ new Vector3();

function closestPointToPoint(
	bvh,
	point,
	target = { },
	minThreshold = 0,
	maxThreshold = Infinity,
) {

	// early out if under minThreshold
	// skip checking if over maxThreshold
	// set minThreshold = maxThreshold to quickly check if a point is within a threshold
	// returns Infinity if no value found
	const minThresholdSq = minThreshold * minThreshold;
	const maxThresholdSq = maxThreshold * maxThreshold;
	let closestDistanceSq = Infinity;
	let closestDistanceTriIndex = null;
	bvh.shapecast(

		{

			boundsTraverseOrder: box => {

				temp.copy( point ).clamp( box.min, box.max );
				return temp.distanceToSquared( point );

			},

			intersectsBounds: ( box, isLeaf, score ) => {

				return score < closestDistanceSq && score < maxThresholdSq;

			},

			intersectsTriangle: ( tri, triIndex ) => {

				tri.closestPointToPoint( point, temp );
				const distSq = point.distanceToSquared( temp );
				if ( distSq < closestDistanceSq ) {

					temp1$2.copy( temp );
					closestDistanceSq = distSq;
					closestDistanceTriIndex = triIndex;

				}

				if ( distSq < minThresholdSq ) {

					return true;

				} else {

					return false;

				}

			},

		}

	);

	if ( closestDistanceSq === Infinity ) return null;

	const closestDistance = Math.sqrt( closestDistanceSq );

	if ( ! target.point ) target.point = temp1$2.clone();
	else target.point.copy( temp1$2 );
	target.distance = closestDistance,
	target.faceIndex = closestDistanceTriIndex;

	return target;

}

const IS_GT_REVISION_169$1 = parseInt( REVISION ) >= 169;
const IS_LT_REVISION_161$1 = parseInt( REVISION ) <= 161;

// Ripped and modified From THREE.js Mesh raycast
// https://github.com/mrdoob/three.js/blob/0aa87c999fe61e216c1133fba7a95772b503eddf/src/objects/Mesh.js#L115
const _vA = /* @__PURE__ */ new Vector3();
const _vB = /* @__PURE__ */ new Vector3();
const _vC = /* @__PURE__ */ new Vector3();

const _uvA$1 = /* @__PURE__ */ new Vector2();
const _uvB$1 = /* @__PURE__ */ new Vector2();
const _uvC$1 = /* @__PURE__ */ new Vector2();

const _normalA$1 = /* @__PURE__ */ new Vector3();
const _normalB$1 = /* @__PURE__ */ new Vector3();
const _normalC$1 = /* @__PURE__ */ new Vector3();

const _intersectionPoint = /* @__PURE__ */ new Vector3();
function checkIntersection( ray, pA, pB, pC, point, side, near, far ) {

	let intersect;
	if ( side === BackSide ) {

		intersect = ray.intersectTriangle( pC, pB, pA, true, point );

	} else {

		intersect = ray.intersectTriangle( pA, pB, pC, side !== DoubleSide, point );

	}

	if ( intersect === null ) return null;

	const distance = ray.origin.distanceTo( point );

	if ( distance < near || distance > far ) return null;

	return {

		distance: distance,
		point: point.clone(),

	};

}

function checkBufferGeometryIntersection( ray, position, normal, uv, uv1, a, b, c, side, near, far ) {

	_vA.fromBufferAttribute( position, a );
	_vB.fromBufferAttribute( position, b );
	_vC.fromBufferAttribute( position, c );

	const intersection = checkIntersection( ray, _vA, _vB, _vC, _intersectionPoint, side, near, far );

	if ( intersection ) {

		if ( uv ) {

			_uvA$1.fromBufferAttribute( uv, a );
			_uvB$1.fromBufferAttribute( uv, b );
			_uvC$1.fromBufferAttribute( uv, c );

			intersection.uv = new Vector2();
			const res = Triangle.getInterpolation( _intersectionPoint, _vA, _vB, _vC, _uvA$1, _uvB$1, _uvC$1, intersection.uv );
			if ( ! IS_GT_REVISION_169$1 ) {

				intersection.uv = res;

			}

		}

		if ( uv1 ) {

			_uvA$1.fromBufferAttribute( uv1, a );
			_uvB$1.fromBufferAttribute( uv1, b );
			_uvC$1.fromBufferAttribute( uv1, c );

			intersection.uv1 = new Vector2();
			const res = Triangle.getInterpolation( _intersectionPoint, _vA, _vB, _vC, _uvA$1, _uvB$1, _uvC$1, intersection.uv1 );
			if ( ! IS_GT_REVISION_169$1 ) {

				intersection.uv1 = res;

			}

			if ( IS_LT_REVISION_161$1 ) {

				intersection.uv2 = intersection.uv1;

			}

		}

		if ( normal ) {

			_normalA$1.fromBufferAttribute( normal, a );
			_normalB$1.fromBufferAttribute( normal, b );
			_normalC$1.fromBufferAttribute( normal, c );

			intersection.normal = new Vector3();
			const res = Triangle.getInterpolation( _intersectionPoint, _vA, _vB, _vC, _normalA$1, _normalB$1, _normalC$1, intersection.normal );
			if ( intersection.normal.dot( ray.direction ) > 0 ) {

				intersection.normal.multiplyScalar( - 1 );

			}

			if ( ! IS_GT_REVISION_169$1 ) {

				intersection.normal = res;

			}

		}

		const face = {
			a: a,
			b: b,
			c: c,
			normal: new Vector3(),
			materialIndex: 0
		};

		Triangle.getNormal( _vA, _vB, _vC, face.normal );

		intersection.face = face;
		intersection.faceIndex = a;

		if ( IS_GT_REVISION_169$1 ) {

			const barycoord = new Vector3();
			Triangle.getBarycoord( _intersectionPoint, _vA, _vB, _vC, barycoord );

			intersection.barycoord = barycoord;

		}

	}

	return intersection;

}

function getSide( materialOrSide ) {

	return materialOrSide && materialOrSide.isMaterial ? materialOrSide.side : materialOrSide;

}

// https://github.com/mrdoob/three.js/blob/0aa87c999fe61e216c1133fba7a95772b503eddf/src/objects/Mesh.js#L258
function intersectTri( geometry, materialOrSide, ray, tri, intersections, near, far ) {

	const triOffset = tri * 3;
	let a = triOffset + 0;
	let b = triOffset + 1;
	let c = triOffset + 2;

	const { index, groups } = geometry;
	if ( geometry.index ) {

		a = index.getX( a );
		b = index.getX( b );
		c = index.getX( c );

	}

	const { position, normal, uv, uv1 } = geometry.attributes;
	if ( Array.isArray( materialOrSide ) ) {

		// check which groups a triangle is present in and run the intersections
		// TODO: we shouldn't need to run and intersection test multiple times
		const firstIndex = tri * 3;
		for ( let i = 0, l = groups.length; i < l; i ++ ) {

			const { start, count, materialIndex } = groups[ i ];
			if ( firstIndex >= start && firstIndex < start + count ) {

				const side = getSide( materialOrSide[ materialIndex ] );
				const intersection = checkBufferGeometryIntersection( ray, position, normal, uv, uv1, a, b, c, side, near, far );
				if ( intersection ) {

					intersection.faceIndex = tri;
					intersection.face.materialIndex = materialIndex;

					if ( intersections ) {

						intersections.push( intersection );

					} else {

						return intersection;

					}

				}

			}

		}

	} else {

		// run the intersection for the single material
		const side = getSide( materialOrSide );
		const intersection = checkBufferGeometryIntersection( ray, position, normal, uv, uv1, a, b, c, side, near, far );
		if ( intersection ) {

			intersection.faceIndex = tri;
			intersection.face.materialIndex = 0;

			if ( intersections ) {

				intersections.push( intersection );

			} else {

				return intersection;

			}

		}

	}

	return null;

}

/** @import { BufferGeometry } from 'three' */

// sets the vertices of triangle `tri` with the 3 vertices after i
function setTriangle( tri, i, index, pos ) {

	const ta = tri.a;
	const tb = tri.b;
	const tc = tri.c;

	let i0 = i;
	let i1 = i + 1;
	let i2 = i + 2;
	if ( index ) {

		i0 = index.getX( i0 );
		i1 = index.getX( i1 );
		i2 = index.getX( i2 );

	}

	ta.x = pos.getX( i0 );
	ta.y = pos.getY( i0 );
	ta.z = pos.getZ( i0 );

	tb.x = pos.getX( i1 );
	tb.y = pos.getY( i1 );
	tb.z = pos.getZ( i1 );

	tc.x = pos.getX( i2 );
	tc.y = pos.getY( i2 );
	tc.z = pos.getZ( i2 );

}

const tempV1 = /* @__PURE__ */ new Vector3();
const tempV2 = /* @__PURE__ */ new Vector3();
const tempV3 = /* @__PURE__ */ new Vector3();
const tempUV1 = /* @__PURE__ */ new Vector2();
const tempUV2 = /* @__PURE__ */ new Vector2();
const tempUV3 = /* @__PURE__ */ new Vector2();

/**
 * @typedef {Object} HitTriangleInfo
 * @property {{ a: number, b: number, c: number, materialIndex: number, normal: Vector3 }} face
 *   Triangle vertex indices, material index, and face normal.
 * @property {Vector2|null} uv - UV coordinates at the hit point, or `null` if no UV attribute is present.
 * @property {Vector3} barycoord - Barycentric coordinates of the hit point within the triangle.
 */

/**
 * Computes hit-point information for a point on a triangle within a `BufferGeometry`. Returns
 * the face vertex indices, face normal, material index, UV coordinates, and barycentric coordinates.
 * Useful for retrieving detailed hit data after a call to `MeshBVH.closestPointToPoint` or
 * `MeshBVH.closestPointToGeometry`.
 *
 * @section Functions
 * @param {Vector3} point - The point on the triangle surface (in the geometry's local space).
 * @param {BufferGeometry} geometry - The geometry containing the triangle.
 * @param {number} triangleIndex - The index of the triangle within the geometry.
 * @param {HitTriangleInfo} [target] - Optional object to write results into. Reuses existing
 *   `face`, `uv`, and `barycoord` sub-objects if present.
 * @returns {HitTriangleInfo}
 */
function getTriangleHitPointInfo( point, geometry, triangleIndex, target ) {

	const indices = geometry.getIndex().array;
	const positions = geometry.getAttribute( 'position' );
	const uvs = geometry.getAttribute( 'uv' );

	const a = indices[ triangleIndex * 3 ];
	const b = indices[ triangleIndex * 3 + 1 ];
	const c = indices[ triangleIndex * 3 + 2 ];

	tempV1.fromBufferAttribute( positions, a );
	tempV2.fromBufferAttribute( positions, b );
	tempV3.fromBufferAttribute( positions, c );

	// find the associated material index
	let materialIndex = 0;
	const groups = geometry.groups;
	const firstVertexIndex = triangleIndex * 3;
	for ( let i = 0, l = groups.length; i < l; i ++ ) {

		const group = groups[ i ];
		const { start, count } = group;
		if ( firstVertexIndex >= start && firstVertexIndex < start + count ) {

			materialIndex = group.materialIndex;
			break;

		}

	}

	// extract barycoord
	const barycoord = target && target.barycoord ? target.barycoord : new Vector3();
	Triangle.getBarycoord( point, tempV1, tempV2, tempV3, barycoord );

	// extract uvs
	let uv = null;
	if ( uvs ) {

		tempUV1.fromBufferAttribute( uvs, a );
		tempUV2.fromBufferAttribute( uvs, b );
		tempUV3.fromBufferAttribute( uvs, c );

		if ( target && target.uv ) uv = target.uv;
		else uv = new Vector2();

		Triangle.getInterpolation( point, tempV1, tempV2, tempV3, tempUV1, tempUV2, tempUV3, uv );

	}

	// adjust the provided target or create a new one
	if ( target ) {

		if ( ! target.face ) target.face = { };
		target.face.a = a;
		target.face.b = b;
		target.face.c = c;
		target.face.materialIndex = materialIndex;
		if ( ! target.face.normal ) target.face.normal = new Vector3();
		Triangle.getNormal( tempV1, tempV2, tempV3, target.face.normal );

		if ( uv ) target.uv = uv;
		target.barycoord = barycoord;

		return target;

	} else {

		return {
			face: {
				a: a,
				b: b,
				c: c,
				materialIndex: materialIndex,
				normal: Triangle.getNormal( tempV1, tempV2, tempV3, new Vector3() )
			},
			uv: uv,
			barycoord: barycoord,
		};

	}

}

/*************************************************************/
/* This file is generated from "iterationUtils.template.js". */
/*************************************************************/

function intersectTris( bvh, materialOrSide, ray, offset, count, intersections, near, far ) {

	const { geometry, _indirectBuffer } = bvh;
	for ( let i = offset, end = offset + count; i < end; i ++ ) {


		intersectTri( geometry, materialOrSide, ray, i, intersections, near, far );


	}

}

function intersectClosestTri( bvh, materialOrSide, ray, offset, count, near, far ) {

	const { geometry, _indirectBuffer } = bvh;
	let dist = Infinity;
	let res = null;
	for ( let i = offset, end = offset + count; i < end; i ++ ) {

		let intersection;

		intersection = intersectTri( geometry, materialOrSide, ray, i, null, near, far );


		if ( intersection && intersection.distance < dist ) {

			res = intersection;
			dist = intersection.distance;

		}

	}

	return res;

}

function iterateOverTriangles$1(
	offset,
	count,
	bvh,
	intersectsTriangleFunc,
	contained,
	depth,
	triangle
) {

	const { geometry } = bvh;
	const { index } = geometry;
	const pos = geometry.attributes.position;
	for ( let i = offset, l = count + offset; i < l; i ++ ) {

		let tri;

		tri = i;

		setTriangle( triangle, tri * 3, index, pos );
		triangle.needsUpdate = true;

		if ( intersectsTriangleFunc( triangle, tri, contained, depth ) ) {

			return true;

		}

	}

	return false;

}

/****************************************************/
/* This file is generated from "refit.template.js". */
/****************************************************/

function refit( bvh, nodeIndices = null ) {

	if ( nodeIndices && Array.isArray( nodeIndices ) ) {

		nodeIndices = new Set( nodeIndices );

	}

	const geometry = bvh.geometry;
	const indexArr = geometry.index ? geometry.index.array : null;
	const posAttr = geometry.attributes.position;

	let buffer, uint32Array, uint16Array, float32Array;
	let byteOffset = 0;
	const roots = bvh._roots;
	for ( let i = 0, l = roots.length; i < l; i ++ ) {

		buffer = roots[ i ];
		uint32Array = new Uint32Array( buffer );
		uint16Array = new Uint16Array( buffer );
		float32Array = new Float32Array( buffer );

		_traverse( 0, byteOffset );
		byteOffset += buffer.byteLength;

	}

	function _traverse( nodeIndex32, byteOffset, force = false ) {

		const nodeIndex16 = nodeIndex32 * 2;
		if ( IS_LEAF( nodeIndex16, uint16Array ) ) {

			const offset = OFFSET( nodeIndex32, uint32Array );
			const count = COUNT( nodeIndex16, uint16Array );

			let minx = Infinity;
			let miny = Infinity;
			let minz = Infinity;
			let maxx = - Infinity;
			let maxy = - Infinity;
			let maxz = - Infinity;


			for ( let i = 3 * offset, l = 3 * ( offset + count ); i < l; i ++ ) {

				let index = indexArr[ i ];
				const x = posAttr.getX( index );
				const y = posAttr.getY( index );
				const z = posAttr.getZ( index );

				if ( x < minx ) minx = x;
				if ( x > maxx ) maxx = x;

				if ( y < miny ) miny = y;
				if ( y > maxy ) maxy = y;

				if ( z < minz ) minz = z;
				if ( z > maxz ) maxz = z;

			}


			if (
				float32Array[ nodeIndex32 + 0 ] !== minx ||
				float32Array[ nodeIndex32 + 1 ] !== miny ||
				float32Array[ nodeIndex32 + 2 ] !== minz ||

				float32Array[ nodeIndex32 + 3 ] !== maxx ||
				float32Array[ nodeIndex32 + 4 ] !== maxy ||
				float32Array[ nodeIndex32 + 5 ] !== maxz
			) {

				float32Array[ nodeIndex32 + 0 ] = minx;
				float32Array[ nodeIndex32 + 1 ] = miny;
				float32Array[ nodeIndex32 + 2 ] = minz;

				float32Array[ nodeIndex32 + 3 ] = maxx;
				float32Array[ nodeIndex32 + 4 ] = maxy;
				float32Array[ nodeIndex32 + 5 ] = maxz;

				return true;

			} else {

				return false;

			}

		} else {

			const left = LEFT_NODE( nodeIndex32 );
			const right = RIGHT_NODE( nodeIndex32, uint32Array );

			// the identifying node indices provided by the shapecast function include offsets of all
			// root buffers to guarantee they're unique between roots so offset left and right indices here.
			let forceChildren = force;
			let includesLeft = false;
			let includesRight = false;

			if ( nodeIndices ) {

				// if we see that neither the left or right child are included in the set that need to be updated
				// then we assume that all children need to be updated.
				if ( ! forceChildren ) {

					const leftNodeId = left / UINT32_PER_NODE + byteOffset / BYTES_PER_NODE;
					const rightNodeId = right / UINT32_PER_NODE + byteOffset / BYTES_PER_NODE;
					includesLeft = nodeIndices.has( leftNodeId );
					includesRight = nodeIndices.has( rightNodeId );
					forceChildren = ! includesLeft && ! includesRight;

				}

			} else {

				includesLeft = true;
				includesRight = true;

			}

			const traverseLeft = forceChildren || includesLeft;
			const traverseRight = forceChildren || includesRight;

			let leftChange = false;
			if ( traverseLeft ) {

				leftChange = _traverse( left, byteOffset, forceChildren );

			}

			let rightChange = false;
			if ( traverseRight ) {

				rightChange = _traverse( right, byteOffset, forceChildren );

			}

			const didChange = leftChange || rightChange;
			if ( didChange ) {

				for ( let i = 0; i < 3; i ++ ) {

					const left_i = left + i;
					const right_i = right + i;
					const minLeftValue = float32Array[ left_i ];
					const maxLeftValue = float32Array[ left_i + 3 ];
					const minRightValue = float32Array[ right_i ];
					const maxRightValue = float32Array[ right_i + 3 ];

					float32Array[ nodeIndex32 + i ] = minLeftValue < minRightValue ? minLeftValue : minRightValue;
					float32Array[ nodeIndex32 + i + 3 ] = maxLeftValue > maxRightValue ? maxLeftValue : maxRightValue;

				}

			}

			return didChange;

		}

	}

}

function intersectsNodeBounds( nodeIndex32, array, ray, near, far ) {

	// This function performs intersection tests similar to Ray.intersectBox in three.js,
	// with the difference that the box values are read from an array to improve performance.

	let tmin, tmax, tymin, tymax, tzmin, tzmax;

	const invdirx = 1 / ray.direction.x,
		invdiry = 1 / ray.direction.y,
		invdirz = 1 / ray.direction.z;

	const ox = ray.origin.x;
	const oy = ray.origin.y;
	const oz = ray.origin.z;

	let minx = array[ nodeIndex32 ];
	let maxx = array[ nodeIndex32 + 3 ];

	let miny = array[ nodeIndex32 + 1 ];
	let maxy = array[ nodeIndex32 + 3 + 1 ];

	let minz = array[ nodeIndex32 + 2 ];
	let maxz = array[ nodeIndex32 + 3 + 2 ];

	if ( invdirx >= 0 ) {

		tmin = ( minx - ox ) * invdirx;
		tmax = ( maxx - ox ) * invdirx;

	} else {

		tmin = ( maxx - ox ) * invdirx;
		tmax = ( minx - ox ) * invdirx;

	}

	if ( invdiry >= 0 ) {

		tymin = ( miny - oy ) * invdiry;
		tymax = ( maxy - oy ) * invdiry;

	} else {

		tymin = ( maxy - oy ) * invdiry;
		tymax = ( miny - oy ) * invdiry;

	}

	if ( ( tmin > tymax ) || ( tymin > tmax ) ) return false;

	if ( tymin > tmin || isNaN( tmin ) ) tmin = tymin;

	if ( tymax < tmax || isNaN( tmax ) ) tmax = tymax;

	if ( invdirz >= 0 ) {

		tzmin = ( minz - oz ) * invdirz;
		tzmax = ( maxz - oz ) * invdirz;

	} else {

		tzmin = ( maxz - oz ) * invdirz;
		tzmax = ( minz - oz ) * invdirz;

	}

	if ( ( tmin > tzmax ) || ( tzmin > tmax ) ) return false;

	if ( tzmin > tmin || tmin !== tmin ) tmin = tzmin;

	if ( tzmax < tmax || tmax !== tmax ) tmax = tzmax;

	//return point closest to the ray (positive side)

	return tmin <= far && tmax >= near;

}

/*************************************************************/
/* This file is generated from "iterationUtils.template.js". */
/*************************************************************/

function intersectTris_indirect( bvh, materialOrSide, ray, offset, count, intersections, near, far ) {

	const { geometry, _indirectBuffer } = bvh;
	for ( let i = offset, end = offset + count; i < end; i ++ ) {

		let vi = _indirectBuffer ? _indirectBuffer[ i ] : i;
		intersectTri( geometry, materialOrSide, ray, vi, intersections, near, far );


	}

}

function intersectClosestTri_indirect( bvh, materialOrSide, ray, offset, count, near, far ) {

	const { geometry, _indirectBuffer } = bvh;
	let dist = Infinity;
	let res = null;
	for ( let i = offset, end = offset + count; i < end; i ++ ) {

		let intersection;
		intersection = intersectTri( geometry, materialOrSide, ray, _indirectBuffer ? _indirectBuffer[ i ] : i, null, near, far );


		if ( intersection && intersection.distance < dist ) {

			res = intersection;
			dist = intersection.distance;

		}

	}

	return res;

}

function iterateOverTriangles_indirect(
	offset,
	count,
	bvh,
	intersectsTriangleFunc,
	contained,
	depth,
	triangle
) {

	const { geometry } = bvh;
	const { index } = geometry;
	const pos = geometry.attributes.position;
	for ( let i = offset, l = count + offset; i < l; i ++ ) {

		let tri;
		tri = bvh.resolveTriangleIndex( i );

		setTriangle( triangle, tri * 3, index, pos );
		triangle.needsUpdate = true;

		if ( intersectsTriangleFunc( triangle, tri, contained, depth ) ) {

			return true;

		}

	}

	return false;

}

/******************************************************/
/* This file is generated from "raycast.template.js". */
/******************************************************/

function raycast( bvh, root, materialOrSide, ray, intersects, near, far ) {

	BufferStack.setBuffer( bvh._roots[ root ] );
	_raycast$1( 0, bvh, materialOrSide, ray, intersects, near, far );
	BufferStack.clearBuffer();

}

function _raycast$1( nodeIndex32, bvh, materialOrSide, ray, intersects, near, far ) {

	const { float32Array, uint16Array, uint32Array } = BufferStack;
	const nodeIndex16 = nodeIndex32 * 2;
	const isLeaf = IS_LEAF( nodeIndex16, uint16Array );
	if ( isLeaf ) {

		const offset = OFFSET( nodeIndex32, uint32Array );
		const count = COUNT( nodeIndex16, uint16Array );


		intersectTris( bvh, materialOrSide, ray, offset, count, intersects, near, far );


	} else {

		const leftIndex = LEFT_NODE( nodeIndex32 );
		if ( intersectsNodeBounds( leftIndex, float32Array, ray, near, far ) ) {

			_raycast$1( leftIndex, bvh, materialOrSide, ray, intersects, near, far );

		}

		const rightIndex = RIGHT_NODE( nodeIndex32, uint32Array );
		if ( intersectsNodeBounds( rightIndex, float32Array, ray, near, far ) ) {

			_raycast$1( rightIndex, bvh, materialOrSide, ray, intersects, near, far );

		}

	}

}

/***********************************************************/
/* This file is generated from "raycastFirst.template.js". */
/***********************************************************/

const _xyzFields$1 = [ 'x', 'y', 'z' ];

function raycastFirst( bvh, root, materialOrSide, ray, near, far ) {

	BufferStack.setBuffer( bvh._roots[ root ] );
	const result = _raycastFirst$1( 0, bvh, materialOrSide, ray, near, far );
	BufferStack.clearBuffer();

	return result;

}

function _raycastFirst$1( nodeIndex32, bvh, materialOrSide, ray, near, far ) {

	const { float32Array, uint16Array, uint32Array } = BufferStack;
	let nodeIndex16 = nodeIndex32 * 2;

	const isLeaf = IS_LEAF( nodeIndex16, uint16Array );
	if ( isLeaf ) {

		const offset = OFFSET( nodeIndex32, uint32Array );
		const count = COUNT( nodeIndex16, uint16Array );


		// eslint-disable-next-line no-unreachable
		return intersectClosestTri( bvh, materialOrSide, ray, offset, count, near, far );


	} else {

		// consider the position of the split plane with respect to the oncoming ray; whichever direction
		// the ray is coming from, look for an intersection among that side of the tree first
		const splitAxis = SPLIT_AXIS( nodeIndex32, uint32Array );
		const xyzAxis = _xyzFields$1[ splitAxis ];
		const rayDir = ray.direction[ xyzAxis ];
		const leftToRight = rayDir >= 0;

		// c1 is the child to check first
		let c1, c2;
		if ( leftToRight ) {

			c1 = LEFT_NODE( nodeIndex32 );
			c2 = RIGHT_NODE( nodeIndex32, uint32Array );

		} else {

			c1 = RIGHT_NODE( nodeIndex32, uint32Array );
			c2 = LEFT_NODE( nodeIndex32 );

		}

		const c1Intersection = intersectsNodeBounds( c1, float32Array, ray, near, far );
		const c1Result = c1Intersection ? _raycastFirst$1( c1, bvh, materialOrSide, ray, near, far ) : null;

		// if we got an intersection in the first node and it's closer than the second node's bounding
		// box, we don't need to consider the second node because it couldn't possibly be a better result
		if ( c1Result ) {

			// check if the point is within the second bounds
			// "point" is in the local frame of the bvh
			const point = c1Result.point[ xyzAxis ];
			const isOutside = leftToRight ?
				point <= float32Array[ c2 + splitAxis ] : // min bounding data
				point >= float32Array[ c2 + splitAxis + 3 ]; // max bounding data

			if ( isOutside ) {

				return c1Result;

			}

		}

		// either there was no intersection in the first node, or there could still be a closer
		// intersection in the second, so check the second node and then take the better of the two
		const c2Intersection = intersectsNodeBounds( c2, float32Array, ray, near, far );
		const c2Result = c2Intersection ? _raycastFirst$1( c2, bvh, materialOrSide, ray, near, far ) : null;

		if ( c1Result && c2Result ) {

			return c1Result.distance <= c2Result.distance ? c1Result : c2Result;

		} else {

			return c1Result || c2Result || null;

		}

	}

}

/*****************************************************************/
/* This file is generated from "intersectsGeometry.template.js". */
/*****************************************************************/
/* eslint-disable indent */

const boundingBox$2 = /* @__PURE__ */ new Box3();
const triangle$1 = /* @__PURE__ */ new ExtendedTriangle();
const triangle2$1 = /* @__PURE__ */ new ExtendedTriangle();
const invertedMat$1 = /* @__PURE__ */ new Matrix4();

const obb$3 = /* @__PURE__ */ new OrientedBox();
const obb2$3 = /* @__PURE__ */ new OrientedBox();

function intersectsGeometry( bvh, root, otherGeometry, geometryToBvh ) {

	BufferStack.setBuffer( bvh._roots[ root ] );
	const result = _intersectsGeometry$1( 0, bvh, otherGeometry, geometryToBvh );
	BufferStack.clearBuffer();

	return result;

}

function _intersectsGeometry$1( nodeIndex32, bvh, otherGeometry, geometryToBvh, cachedObb = null ) {

	const { float32Array, uint16Array, uint32Array } = BufferStack;
	let nodeIndex16 = nodeIndex32 * 2;

	if ( cachedObb === null ) {

		if ( ! otherGeometry.boundingBox ) {

			otherGeometry.computeBoundingBox();

		}

		obb$3.set( otherGeometry.boundingBox.min, otherGeometry.boundingBox.max, geometryToBvh );
		cachedObb = obb$3;

	}

	const isLeaf = IS_LEAF( nodeIndex16, uint16Array );
	if ( isLeaf ) {

		const thisGeometry = bvh.geometry;
		const thisIndex = thisGeometry.index;
		const thisPos = thisGeometry.attributes.position;

		const otherIndex = otherGeometry.index;
		const otherPos = otherGeometry.attributes.position;

		const offset = OFFSET( nodeIndex32, uint32Array );
		const count = COUNT( nodeIndex16, uint16Array );

		// get the inverse of the geometry matrix so we can transform our triangles into the
		// geometry space we're trying to test. We assume there are fewer triangles being checked
		// here.
		invertedMat$1.copy( geometryToBvh ).invert();

		if ( otherGeometry.boundsTree ) {

			// if there's a bounds tree
			arrayToBox( BOUNDING_DATA_INDEX( nodeIndex32 ), float32Array, obb2$3 );
			obb2$3.matrix.copy( invertedMat$1 );
			obb2$3.needsUpdate = true;

			// TODO: use a triangle iteration function here
			const res = otherGeometry.boundsTree.shapecast( {

				intersectsBounds: box => obb2$3.intersectsBox( box ),

				intersectsTriangle: tri => {

					tri.a.applyMatrix4( geometryToBvh );
					tri.b.applyMatrix4( geometryToBvh );
					tri.c.applyMatrix4( geometryToBvh );
					tri.needsUpdate = true;


					for ( let i = offset * 3, l = ( count + offset ) * 3; i < l; i += 3 ) {

						// this triangle needs to be transformed into the current BVH coordinate frame
						setTriangle( triangle2$1, i, thisIndex, thisPos );
						triangle2$1.needsUpdate = true;
						if ( tri.intersectsTriangle( triangle2$1 ) ) {

							return true;

						}

					}


					return false;

				}

			} );

			return res;

		} else {

			// if we're just dealing with raw geometry
			const otherTriangleCount = getTriCount( otherGeometry );


			for ( let i = offset * 3, l = ( count + offset ) * 3; i < l; i += 3 ) {

				// this triangle needs to be transformed into the current BVH coordinate frame
				setTriangle( triangle$1, i, thisIndex, thisPos );


				triangle$1.a.applyMatrix4( invertedMat$1 );
				triangle$1.b.applyMatrix4( invertedMat$1 );
				triangle$1.c.applyMatrix4( invertedMat$1 );
				triangle$1.needsUpdate = true;

				for ( let i2 = 0, l2 = otherTriangleCount * 3; i2 < l2; i2 += 3 ) {

					setTriangle( triangle2$1, i2, otherIndex, otherPos );
					triangle2$1.needsUpdate = true;

					if ( triangle$1.intersectsTriangle( triangle2$1 ) ) {

						return true;

					}

				}


			}


		}

	} else {

		const left = LEFT_NODE( nodeIndex32 );
		const right = RIGHT_NODE( nodeIndex32, uint32Array );

		arrayToBox( BOUNDING_DATA_INDEX( left ), float32Array, boundingBox$2 );
		const leftIntersection =
			cachedObb.intersectsBox( boundingBox$2 ) &&
			_intersectsGeometry$1( left, bvh, otherGeometry, geometryToBvh, cachedObb );

		if ( leftIntersection ) return true;

		arrayToBox( BOUNDING_DATA_INDEX( right ), float32Array, boundingBox$2 );
		const rightIntersection =
			cachedObb.intersectsBox( boundingBox$2 ) &&
			_intersectsGeometry$1( right, bvh, otherGeometry, geometryToBvh, cachedObb );

		if ( rightIntersection ) return true;

		return false;

	}

}

/*********************************************************************/
/* This file is generated from "closestPointToGeometry.template.js". */
/*********************************************************************/

const tempMatrix$1 = /* @__PURE__ */ new Matrix4();
const obb$2 = /* @__PURE__ */ new OrientedBox();
const obb2$2 = /* @__PURE__ */ new OrientedBox();
const temp1$1 = /* @__PURE__ */ new Vector3();
const temp2$1 = /* @__PURE__ */ new Vector3();
const temp3$1 = /* @__PURE__ */ new Vector3();
const temp4$1 = /* @__PURE__ */ new Vector3();

function closestPointToGeometry(
	bvh,
	otherGeometry,
	geometryToBvh,
	target1 = { },
	target2 = { },
	minThreshold = 0,
	maxThreshold = Infinity,
) {

	if ( ! otherGeometry.boundingBox ) {

		otherGeometry.computeBoundingBox();

	}

	obb$2.set( otherGeometry.boundingBox.min, otherGeometry.boundingBox.max, geometryToBvh );
	obb$2.needsUpdate = true;

	const geometry = bvh.geometry;
	const pos = geometry.attributes.position;
	const index = geometry.index;
	const otherPos = otherGeometry.attributes.position;
	const otherIndex = otherGeometry.index;
	const triangle = ExtendedTrianglePool.getPrimitive();
	const triangle2 = ExtendedTrianglePool.getPrimitive();

	let tempTarget1 = temp1$1;
	let tempTargetDest1 = temp2$1;
	let tempTarget2 = null;
	let tempTargetDest2 = null;

	if ( target2 ) {

		tempTarget2 = temp3$1;
		tempTargetDest2 = temp4$1;

	}

	let closestDistance = Infinity;
	let closestDistanceTriIndex = null;
	let closestDistanceOtherTriIndex = null;
	tempMatrix$1.copy( geometryToBvh ).invert();
	obb2$2.matrix.copy( tempMatrix$1 );
	bvh.shapecast(
		{

			boundsTraverseOrder: box => {

				return obb$2.distanceToBox( box );

			},

			intersectsBounds: ( box, isLeaf, score ) => {

				if ( score < closestDistance && score < maxThreshold ) {

					// if we know the triangles of this bounds will be intersected next then
					// save the bounds to use during triangle checks.
					if ( isLeaf ) {

						obb2$2.min.copy( box.min );
						obb2$2.max.copy( box.max );
						obb2$2.needsUpdate = true;

					}

					return true;

				}

				return false;

			},

			intersectsRange: ( offset, count ) => {

				if ( otherGeometry.boundsTree ) {

					// if the other geometry has a bvh then use the accelerated path where we use shapecast to find
					// the closest bounds in the other geometry to check.
					const otherBvh = otherGeometry.boundsTree;
					return otherBvh.shapecast( {
						boundsTraverseOrder: box => {

							return obb2$2.distanceToBox( box );

						},

						intersectsBounds: ( box, isLeaf, score ) => {

							return score < closestDistance && score < maxThreshold;

						},

						intersectsRange: ( otherOffset, otherCount ) => {

							for ( let i2 = otherOffset, l2 = otherOffset + otherCount; i2 < l2; i2 ++ ) {


								setTriangle( triangle2, 3 * i2, otherIndex, otherPos );

								triangle2.a.applyMatrix4( geometryToBvh );
								triangle2.b.applyMatrix4( geometryToBvh );
								triangle2.c.applyMatrix4( geometryToBvh );
								triangle2.needsUpdate = true;

								for ( let i = offset, l = offset + count; i < l; i ++ ) {


									setTriangle( triangle, 3 * i, index, pos );

									triangle.needsUpdate = true;

									const dist = triangle.distanceToTriangle( triangle2, tempTarget1, tempTarget2 );
									if ( dist < closestDistance ) {

										tempTargetDest1.copy( tempTarget1 );

										if ( tempTargetDest2 ) {

											tempTargetDest2.copy( tempTarget2 );

										}

										closestDistance = dist;
										closestDistanceTriIndex = i;
										closestDistanceOtherTriIndex = i2;

									}

									// stop traversal if we find a point that's under the given threshold
									if ( dist < minThreshold ) {

										return true;

									}

								}

							}

						},
					} );

				} else {

					// If no bounds tree then we'll just check every triangle.
					const triCount = getTriCount( otherGeometry );
					for ( let i2 = 0, l2 = triCount; i2 < l2; i2 ++ ) {

						setTriangle( triangle2, 3 * i2, otherIndex, otherPos );
						triangle2.a.applyMatrix4( geometryToBvh );
						triangle2.b.applyMatrix4( geometryToBvh );
						triangle2.c.applyMatrix4( geometryToBvh );
						triangle2.needsUpdate = true;

						for ( let i = offset, l = offset + count; i < l; i ++ ) {


							setTriangle( triangle, 3 * i, index, pos );

							triangle.needsUpdate = true;

							const dist = triangle.distanceToTriangle( triangle2, tempTarget1, tempTarget2 );
							if ( dist < closestDistance ) {

								tempTargetDest1.copy( tempTarget1 );

								if ( tempTargetDest2 ) {

									tempTargetDest2.copy( tempTarget2 );

								}

								closestDistance = dist;
								closestDistanceTriIndex = i;
								closestDistanceOtherTriIndex = i2;

							}

							// stop traversal if we find a point that's under the given threshold
							if ( dist < minThreshold ) {

								return true;

							}

						}

					}

				}

			},

		}

	);

	ExtendedTrianglePool.releasePrimitive( triangle );
	ExtendedTrianglePool.releasePrimitive( triangle2 );

	if ( closestDistance === Infinity ) {

		return null;

	}

	if ( ! target1.point ) {

		target1.point = tempTargetDest1.clone();

	} else {

		target1.point.copy( tempTargetDest1 );

	}

	target1.distance = closestDistance,
	target1.faceIndex = closestDistanceTriIndex;

	if ( target2 ) {

		if ( ! target2.point ) target2.point = tempTargetDest2.clone();
		else target2.point.copy( tempTargetDest2 );
		target2.point.applyMatrix4( tempMatrix$1 );
		tempTargetDest1.applyMatrix4( tempMatrix$1 );
		target2.distance = tempTargetDest1.sub( target2.point ).length();
		target2.faceIndex = closestDistanceOtherTriIndex;

	}

	return target1;

}

/****************************************************/
/* This file is generated from "refit.template.js". */
/****************************************************/

function refit_indirect( bvh, nodeIndices = null ) {

	if ( nodeIndices && Array.isArray( nodeIndices ) ) {

		nodeIndices = new Set( nodeIndices );

	}

	const geometry = bvh.geometry;
	const indexArr = geometry.index ? geometry.index.array : null;
	const posAttr = geometry.attributes.position;

	let buffer, uint32Array, uint16Array, float32Array;
	let byteOffset = 0;
	const roots = bvh._roots;
	for ( let i = 0, l = roots.length; i < l; i ++ ) {

		buffer = roots[ i ];
		uint32Array = new Uint32Array( buffer );
		uint16Array = new Uint16Array( buffer );
		float32Array = new Float32Array( buffer );

		_traverse( 0, byteOffset );
		byteOffset += buffer.byteLength;

	}

	function _traverse( nodeIndex32, byteOffset, force = false ) {

		const nodeIndex16 = nodeIndex32 * 2;
		if ( IS_LEAF( nodeIndex16, uint16Array ) ) {

			const offset = OFFSET( nodeIndex32, uint32Array );
			const count = COUNT( nodeIndex16, uint16Array );

			let minx = Infinity;
			let miny = Infinity;
			let minz = Infinity;
			let maxx = - Infinity;
			let maxy = - Infinity;
			let maxz = - Infinity;

			for ( let i = offset, l = offset + count; i < l; i ++ ) {

				const t = 3 * bvh.resolveTriangleIndex( i );
				for ( let j = 0; j < 3; j ++ ) {

					let index = t + j;
					index = indexArr ? indexArr[ index ] : index;

					const x = posAttr.getX( index );
					const y = posAttr.getY( index );
					const z = posAttr.getZ( index );

					if ( x < minx ) minx = x;
					if ( x > maxx ) maxx = x;

					if ( y < miny ) miny = y;
					if ( y > maxy ) maxy = y;

					if ( z < minz ) minz = z;
					if ( z > maxz ) maxz = z;


				}

			}


			if (
				float32Array[ nodeIndex32 + 0 ] !== minx ||
				float32Array[ nodeIndex32 + 1 ] !== miny ||
				float32Array[ nodeIndex32 + 2 ] !== minz ||

				float32Array[ nodeIndex32 + 3 ] !== maxx ||
				float32Array[ nodeIndex32 + 4 ] !== maxy ||
				float32Array[ nodeIndex32 + 5 ] !== maxz
			) {

				float32Array[ nodeIndex32 + 0 ] = minx;
				float32Array[ nodeIndex32 + 1 ] = miny;
				float32Array[ nodeIndex32 + 2 ] = minz;

				float32Array[ nodeIndex32 + 3 ] = maxx;
				float32Array[ nodeIndex32 + 4 ] = maxy;
				float32Array[ nodeIndex32 + 5 ] = maxz;

				return true;

			} else {

				return false;

			}

		} else {

			const left = LEFT_NODE( nodeIndex32 );
			const right = RIGHT_NODE( nodeIndex32, uint32Array );

			// the identifying node indices provided by the shapecast function include offsets of all
			// root buffers to guarantee they're unique between roots so offset left and right indices here.
			let forceChildren = force;
			let includesLeft = false;
			let includesRight = false;

			if ( nodeIndices ) {

				// if we see that neither the left or right child are included in the set that need to be updated
				// then we assume that all children need to be updated.
				if ( ! forceChildren ) {

					const leftNodeId = left / UINT32_PER_NODE + byteOffset / BYTES_PER_NODE;
					const rightNodeId = right / UINT32_PER_NODE + byteOffset / BYTES_PER_NODE;
					includesLeft = nodeIndices.has( leftNodeId );
					includesRight = nodeIndices.has( rightNodeId );
					forceChildren = ! includesLeft && ! includesRight;

				}

			} else {

				includesLeft = true;
				includesRight = true;

			}

			const traverseLeft = forceChildren || includesLeft;
			const traverseRight = forceChildren || includesRight;

			let leftChange = false;
			if ( traverseLeft ) {

				leftChange = _traverse( left, byteOffset, forceChildren );

			}

			let rightChange = false;
			if ( traverseRight ) {

				rightChange = _traverse( right, byteOffset, forceChildren );

			}

			const didChange = leftChange || rightChange;
			if ( didChange ) {

				for ( let i = 0; i < 3; i ++ ) {

					const left_i = left + i;
					const right_i = right + i;
					const minLeftValue = float32Array[ left_i ];
					const maxLeftValue = float32Array[ left_i + 3 ];
					const minRightValue = float32Array[ right_i ];
					const maxRightValue = float32Array[ right_i + 3 ];

					float32Array[ nodeIndex32 + i ] = minLeftValue < minRightValue ? minLeftValue : minRightValue;
					float32Array[ nodeIndex32 + i + 3 ] = maxLeftValue > maxRightValue ? maxLeftValue : maxRightValue;

				}

			}

			return didChange;

		}

	}

}

/******************************************************/
/* This file is generated from "raycast.template.js". */
/******************************************************/

function raycast_indirect( bvh, root, materialOrSide, ray, intersects, near, far ) {

	BufferStack.setBuffer( bvh._roots[ root ] );
	_raycast( 0, bvh, materialOrSide, ray, intersects, near, far );
	BufferStack.clearBuffer();

}

function _raycast( nodeIndex32, bvh, materialOrSide, ray, intersects, near, far ) {

	const { float32Array, uint16Array, uint32Array } = BufferStack;
	const nodeIndex16 = nodeIndex32 * 2;
	const isLeaf = IS_LEAF( nodeIndex16, uint16Array );
	if ( isLeaf ) {

		const offset = OFFSET( nodeIndex32, uint32Array );
		const count = COUNT( nodeIndex16, uint16Array );

		intersectTris_indirect( bvh, materialOrSide, ray, offset, count, intersects, near, far );


	} else {

		const leftIndex = LEFT_NODE( nodeIndex32 );
		if ( intersectsNodeBounds( leftIndex, float32Array, ray, near, far ) ) {

			_raycast( leftIndex, bvh, materialOrSide, ray, intersects, near, far );

		}

		const rightIndex = RIGHT_NODE( nodeIndex32, uint32Array );
		if ( intersectsNodeBounds( rightIndex, float32Array, ray, near, far ) ) {

			_raycast( rightIndex, bvh, materialOrSide, ray, intersects, near, far );

		}

	}

}

/***********************************************************/
/* This file is generated from "raycastFirst.template.js". */
/***********************************************************/

const _xyzFields = [ 'x', 'y', 'z' ];

function raycastFirst_indirect( bvh, root, materialOrSide, ray, near, far ) {

	BufferStack.setBuffer( bvh._roots[ root ] );
	const result = _raycastFirst( 0, bvh, materialOrSide, ray, near, far );
	BufferStack.clearBuffer();

	return result;

}

function _raycastFirst( nodeIndex32, bvh, materialOrSide, ray, near, far ) {

	const { float32Array, uint16Array, uint32Array } = BufferStack;
	let nodeIndex16 = nodeIndex32 * 2;

	const isLeaf = IS_LEAF( nodeIndex16, uint16Array );
	if ( isLeaf ) {

		const offset = OFFSET( nodeIndex32, uint32Array );
		const count = COUNT( nodeIndex16, uint16Array );

		return intersectClosestTri_indirect( bvh, materialOrSide, ray, offset, count, near, far );


	} else {

		// consider the position of the split plane with respect to the oncoming ray; whichever direction
		// the ray is coming from, look for an intersection among that side of the tree first
		const splitAxis = SPLIT_AXIS( nodeIndex32, uint32Array );
		const xyzAxis = _xyzFields[ splitAxis ];
		const rayDir = ray.direction[ xyzAxis ];
		const leftToRight = rayDir >= 0;

		// c1 is the child to check first
		let c1, c2;
		if ( leftToRight ) {

			c1 = LEFT_NODE( nodeIndex32 );
			c2 = RIGHT_NODE( nodeIndex32, uint32Array );

		} else {

			c1 = RIGHT_NODE( nodeIndex32, uint32Array );
			c2 = LEFT_NODE( nodeIndex32 );

		}

		const c1Intersection = intersectsNodeBounds( c1, float32Array, ray, near, far );
		const c1Result = c1Intersection ? _raycastFirst( c1, bvh, materialOrSide, ray, near, far ) : null;

		// if we got an intersection in the first node and it's closer than the second node's bounding
		// box, we don't need to consider the second node because it couldn't possibly be a better result
		if ( c1Result ) {

			// check if the point is within the second bounds
			// "point" is in the local frame of the bvh
			const point = c1Result.point[ xyzAxis ];
			const isOutside = leftToRight ?
				point <= float32Array[ c2 + splitAxis ] : // min bounding data
				point >= float32Array[ c2 + splitAxis + 3 ]; // max bounding data

			if ( isOutside ) {

				return c1Result;

			}

		}

		// either there was no intersection in the first node, or there could still be a closer
		// intersection in the second, so check the second node and then take the better of the two
		const c2Intersection = intersectsNodeBounds( c2, float32Array, ray, near, far );
		const c2Result = c2Intersection ? _raycastFirst( c2, bvh, materialOrSide, ray, near, far ) : null;

		if ( c1Result && c2Result ) {

			return c1Result.distance <= c2Result.distance ? c1Result : c2Result;

		} else {

			return c1Result || c2Result || null;

		}

	}

}

/*****************************************************************/
/* This file is generated from "intersectsGeometry.template.js". */
/*****************************************************************/
/* eslint-disable indent */

const boundingBox$1 = /* @__PURE__ */ new Box3();
const triangle = /* @__PURE__ */ new ExtendedTriangle();
const triangle2 = /* @__PURE__ */ new ExtendedTriangle();
const invertedMat = /* @__PURE__ */ new Matrix4();

const obb$1 = /* @__PURE__ */ new OrientedBox();
const obb2$1 = /* @__PURE__ */ new OrientedBox();

function intersectsGeometry_indirect( bvh, root, otherGeometry, geometryToBvh ) {

	BufferStack.setBuffer( bvh._roots[ root ] );
	const result = _intersectsGeometry( 0, bvh, otherGeometry, geometryToBvh );
	BufferStack.clearBuffer();

	return result;

}

function _intersectsGeometry( nodeIndex32, bvh, otherGeometry, geometryToBvh, cachedObb = null ) {

	const { float32Array, uint16Array, uint32Array } = BufferStack;
	let nodeIndex16 = nodeIndex32 * 2;

	if ( cachedObb === null ) {

		if ( ! otherGeometry.boundingBox ) {

			otherGeometry.computeBoundingBox();

		}

		obb$1.set( otherGeometry.boundingBox.min, otherGeometry.boundingBox.max, geometryToBvh );
		cachedObb = obb$1;

	}

	const isLeaf = IS_LEAF( nodeIndex16, uint16Array );
	if ( isLeaf ) {

		const thisGeometry = bvh.geometry;
		const thisIndex = thisGeometry.index;
		const thisPos = thisGeometry.attributes.position;

		const otherIndex = otherGeometry.index;
		const otherPos = otherGeometry.attributes.position;

		const offset = OFFSET( nodeIndex32, uint32Array );
		const count = COUNT( nodeIndex16, uint16Array );

		// get the inverse of the geometry matrix so we can transform our triangles into the
		// geometry space we're trying to test. We assume there are fewer triangles being checked
		// here.
		invertedMat.copy( geometryToBvh ).invert();

		if ( otherGeometry.boundsTree ) {

			// if there's a bounds tree
			arrayToBox( BOUNDING_DATA_INDEX( nodeIndex32 ), float32Array, obb2$1 );
			obb2$1.matrix.copy( invertedMat );
			obb2$1.needsUpdate = true;

			// TODO: use a triangle iteration function here
			const res = otherGeometry.boundsTree.shapecast( {

				intersectsBounds: box => obb2$1.intersectsBox( box ),

				intersectsTriangle: tri => {

					tri.a.applyMatrix4( geometryToBvh );
					tri.b.applyMatrix4( geometryToBvh );
					tri.c.applyMatrix4( geometryToBvh );
					tri.needsUpdate = true;

					for ( let i = offset, l = count + offset; i < l; i ++ ) {

						// this triangle needs to be transformed into the current BVH coordinate frame
						setTriangle( triangle2, 3 * bvh.resolveTriangleIndex( i ), thisIndex, thisPos );
						triangle2.needsUpdate = true;
						if ( tri.intersectsTriangle( triangle2 ) ) {

							return true;

						}

					}


					return false;

				}

			} );

			return res;

		} else {

			// if we're just dealing with raw geometry
			const otherTriangleCount = getTriCount( otherGeometry );

			for ( let i = offset, l = count + offset; i < l; i ++ ) {

				// this triangle needs to be transformed into the current BVH coordinate frame
				const ti = bvh.resolveTriangleIndex( i );
				setTriangle( triangle, 3 * ti, thisIndex, thisPos );


				triangle.a.applyMatrix4( invertedMat );
				triangle.b.applyMatrix4( invertedMat );
				triangle.c.applyMatrix4( invertedMat );
				triangle.needsUpdate = true;

				for ( let i2 = 0, l2 = otherTriangleCount * 3; i2 < l2; i2 += 3 ) {

					setTriangle( triangle2, i2, otherIndex, otherPos );
					triangle2.needsUpdate = true;

					if ( triangle.intersectsTriangle( triangle2 ) ) {

						return true;

					}

				}

			}


		}

	} else {

		const left = LEFT_NODE( nodeIndex32 );
		const right = RIGHT_NODE( nodeIndex32, uint32Array );

		arrayToBox( BOUNDING_DATA_INDEX( left ), float32Array, boundingBox$1 );
		const leftIntersection =
			cachedObb.intersectsBox( boundingBox$1 ) &&
			_intersectsGeometry( left, bvh, otherGeometry, geometryToBvh, cachedObb );

		if ( leftIntersection ) return true;

		arrayToBox( BOUNDING_DATA_INDEX( right ), float32Array, boundingBox$1 );
		const rightIntersection =
			cachedObb.intersectsBox( boundingBox$1 ) &&
			_intersectsGeometry( right, bvh, otherGeometry, geometryToBvh, cachedObb );

		if ( rightIntersection ) return true;

		return false;

	}

}

/*********************************************************************/
/* This file is generated from "closestPointToGeometry.template.js". */
/*********************************************************************/

const tempMatrix = /* @__PURE__ */ new Matrix4();
const obb = /* @__PURE__ */ new OrientedBox();
const obb2 = /* @__PURE__ */ new OrientedBox();
const temp1 = /* @__PURE__ */ new Vector3();
const temp2 = /* @__PURE__ */ new Vector3();
const temp3 = /* @__PURE__ */ new Vector3();
const temp4 = /* @__PURE__ */ new Vector3();

function closestPointToGeometry_indirect(
	bvh,
	otherGeometry,
	geometryToBvh,
	target1 = { },
	target2 = { },
	minThreshold = 0,
	maxThreshold = Infinity,
) {

	if ( ! otherGeometry.boundingBox ) {

		otherGeometry.computeBoundingBox();

	}

	obb.set( otherGeometry.boundingBox.min, otherGeometry.boundingBox.max, geometryToBvh );
	obb.needsUpdate = true;

	const geometry = bvh.geometry;
	const pos = geometry.attributes.position;
	const index = geometry.index;
	const otherPos = otherGeometry.attributes.position;
	const otherIndex = otherGeometry.index;
	const triangle = ExtendedTrianglePool.getPrimitive();
	const triangle2 = ExtendedTrianglePool.getPrimitive();

	let tempTarget1 = temp1;
	let tempTargetDest1 = temp2;
	let tempTarget2 = null;
	let tempTargetDest2 = null;

	if ( target2 ) {

		tempTarget2 = temp3;
		tempTargetDest2 = temp4;

	}

	let closestDistance = Infinity;
	let closestDistanceTriIndex = null;
	let closestDistanceOtherTriIndex = null;
	tempMatrix.copy( geometryToBvh ).invert();
	obb2.matrix.copy( tempMatrix );
	bvh.shapecast(
		{

			boundsTraverseOrder: box => {

				return obb.distanceToBox( box );

			},

			intersectsBounds: ( box, isLeaf, score ) => {

				if ( score < closestDistance && score < maxThreshold ) {

					// if we know the triangles of this bounds will be intersected next then
					// save the bounds to use during triangle checks.
					if ( isLeaf ) {

						obb2.min.copy( box.min );
						obb2.max.copy( box.max );
						obb2.needsUpdate = true;

					}

					return true;

				}

				return false;

			},

			intersectsRange: ( offset, count ) => {

				if ( otherGeometry.boundsTree ) {

					// if the other geometry has a bvh then use the accelerated path where we use shapecast to find
					// the closest bounds in the other geometry to check.
					const otherBvh = otherGeometry.boundsTree;
					return otherBvh.shapecast( {
						boundsTraverseOrder: box => {

							return obb2.distanceToBox( box );

						},

						intersectsBounds: ( box, isLeaf, score ) => {

							return score < closestDistance && score < maxThreshold;

						},

						intersectsRange: ( otherOffset, otherCount ) => {

							for ( let i2 = otherOffset, l2 = otherOffset + otherCount; i2 < l2; i2 ++ ) {

								const ti2 = otherBvh.resolveTriangleIndex( i2 );
								setTriangle( triangle2, 3 * ti2, otherIndex, otherPos );

								triangle2.a.applyMatrix4( geometryToBvh );
								triangle2.b.applyMatrix4( geometryToBvh );
								triangle2.c.applyMatrix4( geometryToBvh );
								triangle2.needsUpdate = true;

								for ( let i = offset, l = offset + count; i < l; i ++ ) {

									const ti = bvh.resolveTriangleIndex( i );
									setTriangle( triangle, 3 * ti, index, pos );

									triangle.needsUpdate = true;

									const dist = triangle.distanceToTriangle( triangle2, tempTarget1, tempTarget2 );
									if ( dist < closestDistance ) {

										tempTargetDest1.copy( tempTarget1 );

										if ( tempTargetDest2 ) {

											tempTargetDest2.copy( tempTarget2 );

										}

										closestDistance = dist;
										closestDistanceTriIndex = i;
										closestDistanceOtherTriIndex = i2;

									}

									// stop traversal if we find a point that's under the given threshold
									if ( dist < minThreshold ) {

										return true;

									}

								}

							}

						},
					} );

				} else {

					// If no bounds tree then we'll just check every triangle.
					const triCount = getTriCount( otherGeometry );
					for ( let i2 = 0, l2 = triCount; i2 < l2; i2 ++ ) {

						setTriangle( triangle2, 3 * i2, otherIndex, otherPos );
						triangle2.a.applyMatrix4( geometryToBvh );
						triangle2.b.applyMatrix4( geometryToBvh );
						triangle2.c.applyMatrix4( geometryToBvh );
						triangle2.needsUpdate = true;

						for ( let i = offset, l = offset + count; i < l; i ++ ) {

							const ti = bvh.resolveTriangleIndex( i );
							setTriangle( triangle, 3 * ti, index, pos );

							triangle.needsUpdate = true;

							const dist = triangle.distanceToTriangle( triangle2, tempTarget1, tempTarget2 );
							if ( dist < closestDistance ) {

								tempTargetDest1.copy( tempTarget1 );

								if ( tempTargetDest2 ) {

									tempTargetDest2.copy( tempTarget2 );

								}

								closestDistance = dist;
								closestDistanceTriIndex = i;
								closestDistanceOtherTriIndex = i2;

							}

							// stop traversal if we find a point that's under the given threshold
							if ( dist < minThreshold ) {

								return true;

							}

						}

					}

				}

			},

		}

	);

	ExtendedTrianglePool.releasePrimitive( triangle );
	ExtendedTrianglePool.releasePrimitive( triangle2 );

	if ( closestDistance === Infinity ) {

		return null;

	}

	if ( ! target1.point ) {

		target1.point = tempTargetDest1.clone();

	} else {

		target1.point.copy( tempTargetDest1 );

	}

	target1.distance = closestDistance,
	target1.faceIndex = closestDistanceTriIndex;

	if ( target2 ) {

		if ( ! target2.point ) target2.point = tempTargetDest2.clone();
		else target2.point.copy( tempTargetDest2 );
		target2.point.applyMatrix4( tempMatrix );
		tempTargetDest1.applyMatrix4( tempMatrix );
		target2.distance = tempTargetDest1.sub( target2.point ).length();
		target2.faceIndex = closestDistanceOtherTriIndex;

	}

	return target1;

}

// converts the given BVH raycast intersection to align with the three.js raycast
// structure (include object, world space distance and point).
function convertRaycastIntersect( hit, object, raycaster ) {

	if ( hit === null ) {

		return null;

	}

	hit.point.applyMatrix4( object.matrixWorld );
	hit.distance = hit.point.distanceTo( raycaster.ray.origin );
	hit.object = object;

	return hit;

}

/** @import { BufferGeometry, Sphere, Box3, Intersection, Material, Object3D, Raycaster } from 'three' */
/** @import { ExtendedTriangle } from '../math/ExtendedTriangle.js' */
/** @import { IntersectsBoundsCallback, IntersectsRangeCallback, BoundsTraverseOrderCallback, IntersectsRangesCallback } from './BVH.js' */

const _obb = /* @__PURE__ */ new OrientedBox();
const _ray$4 = /* @__PURE__ */ new Ray();
const _direction = /* @__PURE__ */ new Vector3();
const _inverseMatrix$4 = /* @__PURE__ */ new Matrix4();
const _worldScale = /* @__PURE__ */ new Vector3();
const _getters$1 = [ 'getX', 'getY', 'getZ' ];

/**
 * @callback IntersectsTriangleCallback
 * @param {ExtendedTriangle} triangle - The triangle primitive in local space.
 * @param {number} triangleIndex - The index of the triangle in the geometry.
 * @param {boolean} contained - Whether the node bounds are fully contained by the query shape.
 * @param {number} depth - The depth of the node in the tree.
 * @returns {boolean} Return `true` to stop traversal.
 */

/**
 * @callback IntersectsTrianglesCallback
 * @param {ExtendedTriangle} triangle1 - Triangle from this BVH in local space.
 * @param {ExtendedTriangle} triangle2 - Triangle from `otherBvh`, transformed into local space.
 * @param {number} triangleIndex1 - Triangle index in the first geometry.
 * @param {number} triangleIndex2 - Triangle index in the second geometry.
 * @param {number} depth1 - Depth of the node in the first BVH.
 * @param {number} nodeIndex1 - Node index in the first BVH.
 * @param {number} depth2 - Depth of the node in the second BVH.
 * @param {number} nodeIndex2 - Node index in the second BVH.
 * @returns {boolean} Return `true` to stop traversal.
 */

/**
 * Plain-object representation of a `MeshBVH` produced by `MeshBVH.serialize` and
 * consumed by `MeshBVH.deserialize`. Suitable for transfer across WebWorker boundaries
 * or storage, with optional buffer sharing via `SharedArrayBuffer`.
 *
 * @typedef {Object} SerializedBVH
 * @property {Array<ArrayBuffer>} roots - BVH root node buffers.
 * @property {Int32Array|Uint32Array|Uint16Array|null} index - Serialized geometry index buffer.
 * @property {Uint32Array|Uint16Array|null} indirectBuffer - Indirect primitive index buffer, or `null`
 *   if the BVH was not built in indirect mode.
 */

/**
 * @typedef {Object} HitPointInfo
 * @property {Vector3} point - The closest point on the mesh surface.
 * @property {number} distance - Distance from the query point to the closest point.
 * @property {number} faceIndex - Index of the triangle containing the closest point. Can be
 *   passed to `getTriangleHitPointInfo` to retrieve UV, normal, and material index.
 */

/**
 * The MeshBVH generation process modifies the geometry's index bufferAttribute in place to save
 * memory. The BVH construction will use the geometry's boundingBox if it exists or set it if it
 * does not. The BVH will no longer work correctly if the index buffer is modified.
 *
 * Only triangles within the geometry's draw range (or provided `range` option) are included in the
 * BVH. When a geometry has multiple groups, only triangles within the defined group ranges are
 * included. Triangles in gaps between groups are excluded.
 *
 * Note that all query functions expect arguments in local space of the BVH and return results in
 * local space, as well. If world space results are needed they must be transformed into world space
 * using `object.matrixWorld`.
 *
 * @param {BufferGeometry} geometry
 * @param {Object} [options] - Same options as {@link GeometryBVH}.
 * @extends GeometryBVH
 */
class MeshBVH extends GeometryBVH {

	/**
	 * Generates a representation of the complete bounds tree and the geometry index buffer which
	 * can be used to recreate a bounds tree using the `deserialize` function. The `serialize` and
	 * `deserialize` functions can be used to generate a MeshBVH asynchronously in a background web
	 * worker to prevent the main thread from stuttering. The BVH roots buffer stored in the
	 * serialized representation are the same as the ones used by the original BVH so they should
	 * not be modified. If `SharedArrayBuffers` are used then the same BVH memory can be used for
	 * multiple BVH in multiple WebWorkers.
	 *
	 * @static
	 * @param {MeshBVH} bvh - The BVH to serialize.
	 * @param {Object} [options]
	 * @param {boolean} [options.cloneBuffers=true] - If `true`, the index and BVH root buffers
	 *   are cloned so the serialized data is independent of the live BVH.
	 * @returns {SerializedBVH}
	 */
	static serialize( bvh, options = {} ) {

		options = {
			cloneBuffers: true,
			...options,
		};

		const geometry = bvh.geometry;
		const rootData = bvh._roots;
		const indirectBuffer = bvh._indirectBuffer;
		const indexAttribute = geometry.getIndex();
		const result = {
			version: 1,
			roots: null,
			index: null,
			indirectBuffer: null,
		};
		if ( options.cloneBuffers ) {

			result.roots = rootData.map( root => root.slice() );
			result.index = indexAttribute ? indexAttribute.array.slice() : null;
			result.indirectBuffer = indirectBuffer ? indirectBuffer.slice() : null;

		} else {

			result.roots = rootData;
			result.index = indexAttribute ? indexAttribute.array : null;
			result.indirectBuffer = indirectBuffer;

		}

		return result;

	}

	/**
	 * Returns a new MeshBVH instance from the serialized data. `geometry` is the geometry used
	 * to generate the original BVH `data` was derived from. The root buffers stored in `data`
	 * are set directly on the new BVH so the memory is shared.
	 *
	 * @static
	 * @param {SerializedBVH} data - Serialized BVH data.
	 * @param {BufferGeometry} geometry - The geometry the BVH was originally built from.
	 * @param {Object} [options]
	 * @param {boolean} [options.setIndex=true] - If `true`, sets `geometry.index` from the
	 *   serialized index buffer (creating one if none exists).
	 * @returns {MeshBVH}
	 */
	static deserialize( data, geometry, options = {} ) {

		options = {
			setIndex: true,
			indirect: Boolean( data.indirectBuffer ),
			...options,
		};

		const { index, roots, indirectBuffer } = data;

		// handle backwards compatibility by fixing up the buffer roots
		// see issue gkjohnson/three-mesh-bvh#759
		if ( ! data.version ) {

			console.warn(
				'MeshBVH.deserialize: Serialization format has been changed and will be fixed up. ' +
				'It is recommended to regenerate any stored serialized data.'
			);
			fixupVersion0( roots );

		}

		const bvh = new MeshBVH( geometry, { ...options, [ SKIP_GENERATION ]: true } );
		bvh._roots = roots;
		bvh._indirectBuffer = indirectBuffer || null;

		if ( options.setIndex ) {

			const indexAttribute = geometry.getIndex();
			if ( indexAttribute === null ) {

				const newIndex = new BufferAttribute( data.index, 1, false );
				geometry.setIndex( newIndex );

			} else if ( indexAttribute.array !== index ) {

				indexAttribute.array.set( index );
				indexAttribute.needsUpdate = true;

			}

		}

		return bvh;

		// convert version 0 serialized data (uint32 indices) to version 1 (node indices)
		function fixupVersion0( roots ) {

			for ( let rootIndex = 0; rootIndex < roots.length; rootIndex ++ ) {

				const root = roots[ rootIndex ];
				const uint32Array = new Uint32Array( root );
				const uint16Array = new Uint16Array( root );

				// iterate over nodes and convert right child offsets
				for ( let node = 0, l = root.byteLength / BYTES_PER_NODE; node < l; node ++ ) {

					const node32Index = UINT32_PER_NODE * node;
					const node16Index = 2 * node32Index;
					if ( ! IS_LEAF( node16Index, uint16Array ) ) {

						// convert absolute right child offset to relative offset
						uint32Array[ node32Index + 6 ] = uint32Array[ node32Index + 6 ] / UINT32_PER_NODE - node;

					}

				}

			}

		}

	}

	get primitiveStride() {

		return 3;

	}

	/**
	 * Helper function for use when `indirect` is set to true. This function takes a triangle
	 * index in the BVH layout and returns the associated triangle index in the geometry index
	 * buffer or position attribute.
	 * @type {function(number): number}
	 * @readonly
	 */
	get resolveTriangleIndex() {

		return this.resolvePrimitiveIndex;

	}

	constructor( geometry, options = {} ) {

		if ( options.maxLeafTris ) {

			console.warn( 'MeshBVH: "maxLeafTris" option has been deprecated. Use "targetLeafSize", instead.' );
			options = {
				...options,
				targetLeafSize: options.maxLeafTris,
			};

		}

		super( geometry, options );

	}

	/**
	 * Adjusts all triangle offsets stored in the BVH by the given offset. This is useful when the
	 * triangle data has been compacted or shifted in the geometry buffers (e.g. in `BatchedMesh`
	 * when geometries are compacted using the 'optimize' function or constructing a 'merged' BVH).
	 * This function only adjusts the BVH to point to different triangles in the geometry. The
	 * geometry's index buffer and/or position attributes must be updated separately to match.
	 *
	 * @param {number} offset
	 * @returns {void}
	 */
	// implement abstract methods from BVH base class
	shiftTriangleOffsets( offset ) {

		return super.shiftPrimitiveOffsets( offset );

	}

	// write primitive bounds to the buffer - used only for validateBounds at the moment
	writePrimitiveBounds( i, targetBuffer, baseIndex ) {

		const geometry = this.geometry;
		const indirectBuffer = this._indirectBuffer;
		const posAttr = geometry.attributes.position;
		const index = geometry.index ? geometry.index.array : null;

		const tri = indirectBuffer ? indirectBuffer[ i ] : i;
		const tri3 = tri * 3;

		let ai = tri3 + 0;
		let bi = tri3 + 1;
		let ci = tri3 + 2;

		if ( index ) {

			ai = index[ ai ];
			bi = index[ bi ];
			ci = index[ ci ];

		}

		for ( let el = 0; el < 3; el ++ ) {

			const a = posAttr[ _getters$1[ el ] ]( ai );
			const b = posAttr[ _getters$1[ el ] ]( bi );
			const c = posAttr[ _getters$1[ el ] ]( ci );

			let min = a;
			if ( b < min ) min = b;
			if ( c < min ) min = c;

			let max = a;
			if ( b > max ) max = b;
			if ( c > max ) max = c;

			// Write in min/max format [minx, miny, minz, maxx, maxy, maxz]
			targetBuffer[ baseIndex + el ] = min;
			targetBuffer[ baseIndex + el + 3 ] = max;

		}

		return targetBuffer;

	}

	// precomputes the bounding box for each triangle; required for quickly calculating tree splits.
	// result is an array of size count * 6 where triangle i maps to a
	// [x_center, x_delta, y_center, y_delta, z_center, z_delta] tuple starting at index (i - offset) * 6,
	// representing the center and half-extent in each dimension of triangle i
	computePrimitiveBounds( offset, count, targetBuffer ) {

		const geometry = this.geometry;
		const indirectBuffer = this._indirectBuffer;
		const posAttr = geometry.attributes.position;
		const index = geometry.index ? geometry.index.array : null;
		const normalized = posAttr.normalized;

		if ( offset < 0 || count + offset - targetBuffer.offset > targetBuffer.length / 6 ) {

			throw new Error( 'MeshBVH: compute triangle bounds range is invalid.' );

		}

		// used for non-normalized positions
		const posArr = posAttr.array;

		// support for an interleaved position buffer
		const bufferOffset = posAttr.offset || 0;
		let stride = 3;
		if ( posAttr.isInterleavedBufferAttribute ) {

			stride = posAttr.data.stride;

		}

		// used for normalized positions
		const getters = [ 'getX', 'getY', 'getZ' ];
		const writeOffset = targetBuffer.offset;

		// iterate over the triangle range
		for ( let i = offset, l = offset + count; i < l; i ++ ) {

			const tri = indirectBuffer ? indirectBuffer[ i ] : i;
			const tri3 = tri * 3;
			const boundsIndexOffset = ( i - writeOffset ) * 6;

			let ai = tri3 + 0;
			let bi = tri3 + 1;
			let ci = tri3 + 2;

			if ( index ) {

				ai = index[ ai ];
				bi = index[ bi ];
				ci = index[ ci ];

			}

			// we add the stride and offset here since we access the array directly
			// below for the sake of performance
			if ( ! normalized ) {

				ai = ai * stride + bufferOffset;
				bi = bi * stride + bufferOffset;
				ci = ci * stride + bufferOffset;

			}

			for ( let el = 0; el < 3; el ++ ) {

				let a, b, c;

				if ( normalized ) {

					a = posAttr[ getters[ el ] ]( ai );
					b = posAttr[ getters[ el ] ]( bi );
					c = posAttr[ getters[ el ] ]( ci );

				} else {

					a = posArr[ ai + el ];
					b = posArr[ bi + el ];
					c = posArr[ ci + el ];

				}

				let min = a;
				if ( b < min ) min = b;
				if ( c < min ) min = c;

				let max = a;
				if ( b > max ) max = b;
				if ( c > max ) max = c;

				// Increase the bounds size by float32 epsilon to avoid precision errors when
				// converting to 32 bit float. Scale the epsilon by the size of the numbers being
				// worked with.
				const halfExtents = ( max - min ) / 2;
				const el2 = el * 2;
				targetBuffer[ boundsIndexOffset + el2 + 0 ] = min + halfExtents;
				targetBuffer[ boundsIndexOffset + el2 + 1 ] = halfExtents + ( Math.abs( min ) + halfExtents ) * FLOAT32_EPSILON;

			}

		}

		return targetBuffer;

	}

	/**
	 * A convenience function for performing a raycast based on a mesh. Results are formed like
	 * three.js raycast results in world frame.
	 *
	 * @param {Object3D} object
	 * @param {Raycaster} raycaster
	 * @param {Array<Intersection>} [intersects=[]]
	 * @returns {Array<Intersection>}
	 */
	raycastObject3D( object, raycaster, intersects = [] ) {

		const { material } = object;
		if ( material === undefined ) {

			return;

		}

		_inverseMatrix$4.copy( object.matrixWorld ).invert();
		_ray$4.copy( raycaster.ray ).applyMatrix4( _inverseMatrix$4 );

		_worldScale.setFromMatrixScale( object.matrixWorld );
		_direction.copy( _ray$4.direction ).multiply( _worldScale );

		const scaleFactor = _direction.length();
		const near = raycaster.near / scaleFactor;
		const far = raycaster.far / scaleFactor;

		if ( raycaster.firstHitOnly === true ) {

			let hit = this.raycastFirst( _ray$4, material, near, far );
			hit = convertRaycastIntersect( hit, object, raycaster );
			if ( hit ) {

				intersects.push( hit );

			}

		} else {

			const hits = this.raycast( _ray$4, material, near, far );
			for ( let i = 0, l = hits.length; i < l; i ++ ) {

				const hit = convertRaycastIntersect( hits[ i ], object, raycaster );
				if ( hit ) {

					intersects.push( hit );

				}

			}

		}

		return intersects;

	}

	/**
	 * Refit the node bounds to the current triangle positions. This is quicker than regenerating
	 * a new BVH but will not be optimal after significant changes to the vertices. `nodeIndices`
	 * is a set of node indices (provided by the `shapecast` function) that need to be refit
	 * including all internal nodes.
	 *
	 * @param {Set<number>|Array<number>|null} [nodeIndices=null]
	 */
	refit( nodeIndices = null ) {

		const refitFunc = this.indirect ? refit_indirect : refit;
		return refitFunc( this, nodeIndices );

	}

	/* Core Cast Functions */

	/**
	 * Returns all raycast triangle hits in unsorted order. It is expected that `ray` is in the
	 * frame of the BVH already. Likewise the returned results are also provided in the local
	 * frame of the BVH. The `side` identifier is used to determine the side to check when
	 * raycasting or a material with the given side field can be passed. If an array of materials
	 * is provided then it is expected that the geometry has groups and the appropriate material
	 * side is used per group.
	 *
	 * Note that unlike three.js' Raycaster results the points and distances in the intersections
	 * returned from this function are relative to the local frame of the MeshBVH. When using the
	 * `acceleratedRaycast` function as an override for `Mesh.raycast` they are transformed into
	 * world space to be consistent with three's results.
	 *
	 * @param {Ray} ray
	 * @param {number|Material|Array<Material>} [materialOrSide=FrontSide]
	 * @param {number} [near=0]
	 * @param {number} [far=Infinity]
	 * @returns {Array<Intersection>}
	 */
	raycast( ray, materialOrSide = FrontSide, near = 0, far = Infinity ) {

		const roots = this._roots;
		const intersects = [];
		const raycastFunc = this.indirect ? raycast_indirect : raycast;
		for ( let i = 0, l = roots.length; i < l; i ++ ) {

			raycastFunc( this, i, materialOrSide, ray, intersects, near, far );

		}

		return intersects;

	}

	/**
	 * Returns the first raycast hit in the model. This is typically much faster than returning
	 * all hits. See `raycast` for information on the side and material options as well as the
	 * frame of the returned intersections.
	 *
	 * @param {Ray} ray
	 * @param {number|Material|Array<Material>} [materialOrSide=FrontSide]
	 * @param {number} [near=0]
	 * @param {number} [far=Infinity]
	 * @returns {Intersection|null}
	 */
	raycastFirst( ray, materialOrSide = FrontSide, near = 0, far = Infinity ) {

		const roots = this._roots;
		let closestResult = null;

		const raycastFirstFunc = this.indirect ? raycastFirst_indirect : raycastFirst;
		for ( let i = 0, l = roots.length; i < l; i ++ ) {

			const result = raycastFirstFunc( this, i, materialOrSide, ray, near, far );
			if ( result != null && ( closestResult == null || result.distance < closestResult.distance ) ) {

				closestResult = result;

			}

		}

		return closestResult;

	}

	/**
	 * Returns whether or not the mesh intersects the given geometry.
	 *
	 * The `geometryToBvh` parameter is the transform of the geometry in the BVH's local frame.
	 *
	 * Performance improves considerably if the provided geometry also has a `boundsTree`.
	 *
	 * @param {BufferGeometry} otherGeometry
	 * @param {Matrix4} geometryToBvh - Transform of `otherGeometry` into the local space of
	 *   this BVH.
	 * @returns {boolean}
	 */
	intersectsGeometry( otherGeometry, geomToMesh ) {

		let result = false;
		const roots = this._roots;
		const intersectsGeometryFunc = this.indirect ? intersectsGeometry_indirect : intersectsGeometry;
		for ( let i = 0, l = roots.length; i < l; i ++ ) {

			result = intersectsGeometryFunc( this, i, otherGeometry, geomToMesh );

			if ( result ) {

				break;

			}

		}

		return result;

	}

	/**
	 * A generalized cast function that can be used to implement intersection logic for custom
	 * shapes. This is used internally for `intersectsBox`, `intersectsSphere`, and more. The
	 * function returns as soon as a triangle has been reported as intersected and returns `true`
	 * if a triangle has been intersected.
	 *
	 * @param {Object} callbacks
	 * @param {IntersectsBoundsCallback} callbacks.intersectsBounds
	 * @param {IntersectsTriangleCallback} [callbacks.intersectsTriangle]
	 * @param {IntersectsRangeCallback} [callbacks.intersectsRange]
	 * @param {BoundsTraverseOrderCallback} [callbacks.boundsTraverseOrder]
	 * @returns {boolean}
	 */
	shapecast( callbacks ) {

		const triangle = ExtendedTrianglePool.getPrimitive();
		const result = super.shapecast(
			{
				...callbacks,
				intersectsPrimitive: callbacks.intersectsTriangle,
				scratchPrimitive: triangle,

				// TODO: is the performance significant enough for the added complexity here?
				// can we just use one function?
				iterate: this.indirect ? iterateOverTriangles_indirect : iterateOverTriangles$1,
			}
		);
		ExtendedTrianglePool.releasePrimitive( triangle );

		return result;

	}

	/**
	 * A generalized cast function that traverses two BVH structures simultaneously to perform
	 * intersection tests between them. This is used internally by `intersectsGeometry`. The
	 * function returns `true` as soon as a triangle pair has been reported as intersected by
	 * the callbacks.
	 *
	 * `matrixToLocal` is a Matrix4 that transforms `otherBvh` into the local space of this BVH.
	 * The other BVH's triangles are transformed by this matrix before intersection tests.
	 *
	 * @param {MeshBVH} otherBvh
	 * @param {Matrix4} matrixToLocal - Transforms `otherBvh` into the local space of this BVH.
	 * @param {Object} callbacks
	 * @param {IntersectsRangesCallback} [callbacks.intersectsRanges]
	 * @param {IntersectsTrianglesCallback} [callbacks.intersectsTriangles]
	 * @returns {boolean}
	 */
	bvhcast( otherBvh, matrixToLocal, callbacks ) {

		let {
			intersectsRanges,
			intersectsTriangles,
		} = callbacks;

		const triangle1 = ExtendedTrianglePool.getPrimitive();
		const indexAttr1 = this.geometry.index;
		const positionAttr1 = this.geometry.attributes.position;
		const assignTriangle1 = this.indirect ?
			i1 => {


				const ti = this.resolveTriangleIndex( i1 );
				setTriangle( triangle1, ti * 3, indexAttr1, positionAttr1 );

			} :
			i1 => {

				setTriangle( triangle1, i1 * 3, indexAttr1, positionAttr1 );

			};

		const triangle2 = ExtendedTrianglePool.getPrimitive();
		const indexAttr2 = otherBvh.geometry.index;
		const positionAttr2 = otherBvh.geometry.attributes.position;
		const assignTriangle2 = otherBvh.indirect ?
			i2 => {

				const ti2 = otherBvh.resolveTriangleIndex( i2 );
				setTriangle( triangle2, ti2 * 3, indexAttr2, positionAttr2 );

			} :
			i2 => {

				setTriangle( triangle2, i2 * 3, indexAttr2, positionAttr2 );

			};

		// generate triangle callback if needed
		if ( intersectsTriangles ) {

			if ( ! ( otherBvh instanceof MeshBVH ) ) {

				throw new Error( 'MeshBVH: "intersectsTriangles" callback can only be used with another MeshBVH.' );

			}

			const iterateOverDoubleTriangles = ( offset1, count1, offset2, count2, depth1, nodeIndex1, depth2, nodeIndex2 ) => {

				for ( let i2 = offset2, l2 = offset2 + count2; i2 < l2; i2 ++ ) {

					assignTriangle2( i2 );

					triangle2.a.applyMatrix4( matrixToLocal );
					triangle2.b.applyMatrix4( matrixToLocal );
					triangle2.c.applyMatrix4( matrixToLocal );
					triangle2.needsUpdate = true;

					for ( let i1 = offset1, l1 = offset1 + count1; i1 < l1; i1 ++ ) {

						assignTriangle1( i1 );

						triangle1.needsUpdate = true;

						if ( intersectsTriangles( triangle1, triangle2, i1, i2, depth1, nodeIndex1, depth2, nodeIndex2 ) ) {

							return true;

						}

					}

				}

				return false;

			};

			if ( intersectsRanges ) {

				const originalIntersectsRanges = intersectsRanges;
				intersectsRanges = function ( offset1, count1, offset2, count2, depth1, nodeIndex1, depth2, nodeIndex2 ) {

					if ( ! originalIntersectsRanges( offset1, count1, offset2, count2, depth1, nodeIndex1, depth2, nodeIndex2 ) ) {

						return iterateOverDoubleTriangles( offset1, count1, offset2, count2, depth1, nodeIndex1, depth2, nodeIndex2 );

					}

					return true;

				};

			} else {

				intersectsRanges = iterateOverDoubleTriangles;

			}

		}

		return super.bvhcast( otherBvh, matrixToLocal, { intersectsRanges } );

	}


	/* Derived Cast Functions */

	/**
	 * Returns whether or not the mesh intersects the given box.
	 *
	 * The `boxToBvh` parameter is the transform of the box in the meshes frame.
	 *
	 * @param {Box3} box
	 * @param {Matrix4} boxToBvh - Transform of the box in the local space of this BVH.
	 * @returns {boolean}
	 */
	intersectsBox( box, boxToMesh ) {

		_obb.set( box.min, box.max, boxToMesh );
		_obb.needsUpdate = true;

		return this.shapecast(
			{
				intersectsBounds: box => _obb.intersectsBox( box ),
				intersectsTriangle: tri => _obb.intersectsTriangle( tri )
			}
		);

	}

	/**
	 * Returns whether or not the mesh intersects the given sphere.
	 *
	 * @param {Sphere} sphere
	 * @returns {boolean}
	 */
	intersectsSphere( sphere ) {

		return this.shapecast(
			{
				intersectsBounds: box => sphere.intersectsBox( box ),
				intersectsTriangle: tri => tri.intersectsSphere( sphere )
			}
		);

	}

	/**
	 * Computes the closest distance from the geometry to the mesh and puts the closest point on
	 * the mesh in `target1` (in the frame of the BVH) and the closest point on the other
	 * geometry in `target2` (in the geometry frame). If `target1` is not provided a new Object
	 * is created and returned from the function.
	 *
	 * The `geometryToBvh` parameter is the transform of the geometry in the BVH's local frame.
	 *
	 * If a point is found that is closer than `minThreshold` then the function will return that
	 * result early. Any triangles or points outside of `maxThreshold` are ignored. If no point
	 * is found within the min / max thresholds then `null` is returned and the target objects
	 * are not modified.
	 *
	 * The returned faceIndex in `target1` and `target2` can be used with the standalone function
	 * `getTriangleHitPointInfo` to obtain more information like UV coordinates, triangle normal
	 * and materialIndex.
	 *
	 * _Note that this function can be very slow if `geometry` does not have a
	 * `geometry.boundsTree` computed._
	 *
	 * @param {BufferGeometry} otherGeometry
	 * @param {Matrix4} geometryToBvh - Transform of `otherGeometry` into the local space of
	 *   this BVH.
	 * @param {HitPointInfo} [target1={}]
	 * @param {HitPointInfo} [target2={}]
	 * @param {number} [minThreshold=0]
	 * @param {number} [maxThreshold=Infinity]
	 * @returns {HitPointInfo|null}
	 */
	closestPointToGeometry( otherGeometry, geometryToBvh, target1 = { }, target2 = { }, minThreshold = 0, maxThreshold = Infinity ) {

		const closestPointToGeometryFunc = this.indirect ? closestPointToGeometry_indirect : closestPointToGeometry;
		return closestPointToGeometryFunc(
			this,
			otherGeometry,
			geometryToBvh,
			target1,
			target2,
			minThreshold,
			maxThreshold,
		);

	}

	/**
	 * Computes the closest distance from the point to the mesh and gives additional information
	 * in `target`. The target can be left undefined to default to a new object which is
	 * ultimately returned by the function.
	 *
	 * If a point is found that is closer than `minThreshold` then the function will return that
	 * result early. Any triangles or points outside of `maxThreshold` are ignored. If no point
	 * is found within the min / max thresholds then `null` is returned and the `target` object
	 * is not modified.
	 *
	 * The returned faceIndex can be used with the standalone function `getTriangleHitPointInfo`
	 * to obtain more information like UV coordinates, triangle normal and materialIndex.
	 *
	 * @param {Vector3} point
	 * @param {HitPointInfo} [target={}]
	 * @param {number} [minThreshold=0]
	 * @param {number} [maxThreshold=Infinity]
	 * @returns {HitPointInfo|null}
	 */
	closestPointToPoint( point, target = { }, minThreshold = 0, maxThreshold = Infinity ) {

		return closestPointToPoint(
			this,
			point,
			target,
			minThreshold,
			maxThreshold,
		);

	}

}

/** @import { BufferGeometry } from 'three' */
/** @import { IntersectsBoundsCallback, IntersectsRangeCallback, BoundsTraverseOrderCallback } from './BVH.js' */

const _inverseMatrix$3 = /* @__PURE__ */ new Matrix4();
const _ray$3 = /* @__PURE__ */ new Ray();
const _linePool = /* @__PURE__ */ new PrimitivePool( () => new Line3() );
const _intersectPointOnRay = /*@__PURE__*/ new Vector3();
const _intersectPointOnSegment = /*@__PURE__*/ new Vector3();
const _box$2 = /* @__PURE__ */ new Box3();
const _vec$3 = /* @__PURE__ */ new Vector3();
const _getters = [ 'getX', 'getY', 'getZ' ];

/**
 * @callback IntersectsLineCallback
 * @param {Line3} line - The line segment primitive in local space.
 * @param {number} index - The primitive index within the BVH buffer.
 * @param {boolean} contained - Whether the node bounds are fully contained by the query shape.
 * @param {number} depth - The depth of the node in the tree.
 * @returns {boolean} Return `true` to stop traversal.
 */

/**
 * BVH for `THREE.LineSegments` geometries. Each BVH primitive represents one line segment
 * (two consecutive vertices).
 * @extends GeometryBVH
 */
class LineSegmentsBVH extends GeometryBVH {

	get primitiveStride() {

		return 2;

	}

	writePrimitiveBounds( i, targetBuffer, baseIndex ) {

		const indirectBuffer = this._indirectBuffer;
		const { geometry, primitiveStride } = this;

		const posAttr = geometry.attributes.position;
		const indexAttr = geometry.index;

		// TODO: this may not be right for a LineLoop with a limited draw range / groups
		const vertCount = indexAttr ? indexAttr.count : posAttr.count;

		const prim = indirectBuffer ? indirectBuffer[ i ] : i;
		let i0 = prim * primitiveStride;
		let i1 = ( i0 + 1 ) % vertCount;
		if ( indexAttr ) {

			i0 = indexAttr.getX( i0 );
			i1 = indexAttr.getX( i1 );

		}

		for ( let el = 0; el < 3; el ++ ) {

			const v0 = posAttr[ _getters[ el ] ]( i0 );
			const v1 = posAttr[ _getters[ el ] ]( i1 );
			const min = v0 < v1 ? v0 : v1;
			const max = v0 > v1 ? v0 : v1;

			// Write in min/max format [minx, miny, minz, maxx, maxy, maxz]
			targetBuffer[ baseIndex + el ] = min;
			targetBuffer[ baseIndex + el + 3 ] = max;

		}

		return targetBuffer;

	}

	/**
	 * Performs a spatial query against the BVH. Extends the base `shapecast` with an
	 * `intersectsLine` callback that is called once per line segment primitive in leaf nodes.
	 *
	 * @param {Object} callbacks
	 * @param {IntersectsBoundsCallback} callbacks.intersectsBounds
	 * @param {IntersectsLineCallback} [callbacks.intersectsLine]
	 * @param {IntersectsRangeCallback} [callbacks.intersectsRange]
	 * @param {BoundsTraverseOrderCallback} [callbacks.boundsTraverseOrder]
	 * @returns {boolean}
	 */
	shapecast( callbacks ) {

		const line = _linePool.getPrimitive();
		const result = super.shapecast( {
			...callbacks,
			intersectsPrimitive: callbacks.intersectsLine,
			scratchPrimitive: line,
			iterate: iterateOverLines,
		} );
		_linePool.releasePrimitive( line );

		return result;

	}

	raycastObject3D( object, raycaster, intersects = [] ) {

		const { matrixWorld } = object;
		const { firstHitOnly } = raycaster;

		_inverseMatrix$3.copy( matrixWorld ).invert();
		_ray$3.copy( raycaster.ray ).applyMatrix4( _inverseMatrix$3 );

		const threshold = raycaster.params.Line.threshold;
		const localThreshold = threshold / ( ( object.scale.x + object.scale.y + object.scale.z ) / 3 );
		const localThresholdSq = localThreshold * localThreshold;

		let closestHit = null;
		let closestDistance = Infinity;
		this.shapecast( {
			boundsTraverseOrder: box => {

				return box.distanceToPoint( _ray$3.origin );

			},
			intersectsBounds: box => {

				_box$2.copy( box ).expandByScalar( localThreshold );

				if ( firstHitOnly ) {

					if ( ! _ray$3.intersectBox( _box$2, _vec$3 ) ) {

						return NOT_INTERSECTED;

					}

					let dist;
					if ( _box$2.containsPoint( _ray$3.origin ) ) {

						dist = 0;

					} else {

						_vec$3.applyMatrix4( matrixWorld );
						dist = raycaster.ray.origin.distanceTo( _vec$3 );

					}

					// early out if the box is further than the closest raycast
					return dist < closestDistance ? INTERSECTED : NOT_INTERSECTED;

				} else {

					return _ray$3.intersectsBox( _box$2 ) ? INTERSECTED : NOT_INTERSECTED;

				}

			},
			intersectsLine: ( line, index ) => {

				const distSq = _ray$3.distanceSqToSegment( line.start, line.end, _intersectPointOnRay, _intersectPointOnSegment );

				if ( distSq > localThresholdSq ) return;

				_intersectPointOnRay.applyMatrix4( object.matrixWorld );

				const distance = raycaster.ray.origin.distanceTo( _intersectPointOnRay );

				if ( distance < raycaster.near || distance > raycaster.far ) return;

				if ( firstHitOnly && distance >= closestDistance ) return;
				closestDistance = distance;

				index = this.resolvePrimitiveIndex( index );

				closestHit = {
					distance,
					point: _intersectPointOnSegment.clone().applyMatrix4( matrixWorld ),
					index: index * this.primitiveStride,
					face: null,
					faceIndex: null,
					barycoord: null,
					object,
				};

				if ( ! firstHitOnly ) {

					intersects.push( closestHit );

				}

			},
		} );

		if ( firstHitOnly && closestHit ) {

			intersects.push( closestHit );

		}

		return intersects;

	}

}

/**
 * BVH for `THREE.LineLoop` geometries. Forces indirect mode since the loop structure
 * requires that the index buffer remain unmodified.
 * @param {BufferGeometry} geometry
 * @param {Object} [options] - Same options as {@link GeometryBVH}. `indirect` is always forced to `true`.
 * @extends LineSegmentsBVH
 */
class LineLoopBVH extends LineSegmentsBVH {

	get primitiveStride() {

		return 1;

	}

	constructor( geometry, options = {} ) {

		// "Line" and "LineLoop" BVH must be indirect since we cannot rearrange the index
		// buffer without breaking the lines
		options = {
			...options,
			indirect: true,
		};

		super( geometry, options );

	}

}

/**
 * BVH for `THREE.Line` geometries. Like `LineLoopBVH` but excludes the final closing
 * segment so the open line is accurately represented.
 * @param {BufferGeometry} geometry
 * @param {Object} [options] - Same options as {@link GeometryBVH}. `indirect` is always forced to `true`.
 * @extends LineLoopBVH
 */
class LineBVH extends LineLoopBVH {

	getRootRanges( ...args ) {

		const res = super.getRootRanges( ...args );
		res.forEach( group => group.count -- );
		return res;

	}

}

function iterateOverLines(
	offset,
	count,
	bvh,
	intersectsPointFunc,
	contained,
	depth,
	line
) {

	const { geometry, primitiveStride } = bvh;
	const { index } = geometry;
	const posAttr = geometry.attributes.position;
	const vertCount = index ? index.count : posAttr.count;

	for ( let i = offset, l = count + offset; i < l; i ++ ) {

		const prim = bvh.resolvePrimitiveIndex( i );
		let i0 = prim * primitiveStride;
		let i1 = ( i0 + 1 ) % vertCount;
		if ( index ) {

			i0 = index.getX( i0 );
			i1 = index.getX( i1 );

		}

		line.start.fromBufferAttribute( posAttr, i0 );
		line.end.fromBufferAttribute( posAttr, i1 );

		if ( intersectsPointFunc( line, i, contained, depth ) ) {

			return true;

		}

	}

	return false;

}

/** @import { IntersectsBoundsCallback, IntersectsRangeCallback, BoundsTraverseOrderCallback } from './BVH.js' */

const _inverseMatrix$2 = /* @__PURE__ */ new Matrix4();
const _ray$2 = /* @__PURE__ */ new Ray();
const _pointPool = /* @__PURE__ */ new PrimitivePool( () => new Vector3() );
const _box$1 = /* @__PURE__ */ new Box3();
const _vec$2 = /* @__PURE__ */ new Vector3();

/**
 * @callback IntersectsPointCallback
 * @param {Vector3} point - The point primitive in local space.
 * @param {number} index - The primitive index within the BVH buffer.
 * @param {boolean} contained - Whether the node bounds are fully contained by the query shape.
 * @param {number} depth - The depth of the node in the tree.
 * @returns {boolean} Return `true` to stop traversal.
 */

/**
 * BVH for `THREE.Points` geometries. Each BVH primitive represents a single point.
 * @extends GeometryBVH
 */
class PointsBVH extends GeometryBVH {

	get primitiveStride() {

		return 1;

	}

	writePrimitiveBounds( i, targetBuffer, baseIndex ) {

		const indirectBuffer = this._indirectBuffer;
		const { geometry } = this;
		const posAttr = geometry.attributes.position;
		const indexAttr = geometry.index;
		let pointIndex = indirectBuffer ? indirectBuffer[ i ] : i;
		if ( indexAttr ) {

			pointIndex = indexAttr.getX( pointIndex );

		}

		const px = posAttr.getX( pointIndex );
		const py = posAttr.getY( pointIndex );
		const pz = posAttr.getZ( pointIndex );

		// Write in min/max format [minx, miny, minz, maxx, maxy, maxz]
		// For points, min equals max (epsilon padding is applied in computePrimitiveBounds)
		targetBuffer[ baseIndex + 0 ] = px;
		targetBuffer[ baseIndex + 1 ] = py;
		targetBuffer[ baseIndex + 2 ] = pz;
		targetBuffer[ baseIndex + 3 ] = px;
		targetBuffer[ baseIndex + 4 ] = py;
		targetBuffer[ baseIndex + 5 ] = pz;

		return targetBuffer;

	}

	/**
	 * Performs a spatial query against the BVH. Extends the base `shapecast` with an
	 * `intersectsPoint` callback that is called once per point primitive in leaf nodes.
	 *
	 * @param {Object} callbacks
	 * @param {IntersectsBoundsCallback} callbacks.intersectsBounds
	 * @param {IntersectsPointCallback} [callbacks.intersectsPoint]
	 * @param {IntersectsRangeCallback} [callbacks.intersectsRange]
	 * @param {BoundsTraverseOrderCallback} [callbacks.boundsTraverseOrder]
	 * @returns {boolean}
	 */
	shapecast( callbacks ) {

		// TODO: avoid unnecessary "iterate over points" function
		const point = _pointPool.getPrimitive();
		const result = super.shapecast(
			{
				...callbacks,
				intersectsPrimitive: callbacks.intersectsPoint,
				scratchPrimitive: point,
				iterate: iterateOverPoints,
			},
		);

		_pointPool.releasePrimitive( point );
		return result;

	}

	raycastObject3D( object, raycaster, intersects = [] ) {

		const { geometry } = this;
		const { matrixWorld } = object;
		const { firstHitOnly } = raycaster;

		_inverseMatrix$2.copy( matrixWorld ).invert();
		_ray$2.copy( raycaster.ray ).applyMatrix4( _inverseMatrix$2 );

		const threshold = raycaster.params.Points.threshold;
		const localThreshold = threshold / ( ( object.scale.x + object.scale.y + object.scale.z ) / 3 );
		const localThresholdSq = localThreshold * localThreshold;

		let closestHit = null;
		let closestDistance = Infinity;
		this.shapecast( {
			boundsTraverseOrder: box => {

				return box.distanceToPoint( _ray$2.origin );

			},
			intersectsBounds: box => {

				_box$1.copy( box ).expandByScalar( localThreshold );

				if ( firstHitOnly ) {

					if ( ! _ray$2.intersectBox( _box$1, _vec$2 ) ) {

						return NOT_INTERSECTED;

					}

					let dist;
					if ( _box$1.containsPoint( _ray$2.origin ) ) {

						dist = 0;

					} else {

						_vec$2.applyMatrix4( matrixWorld );
						dist = raycaster.ray.origin.distanceTo( _vec$2 );

					}

					// early out if the box is further than the closest raycast
					return dist < closestDistance ? INTERSECTED : NOT_INTERSECTED;

				} else {

					return _ray$2.intersectsBox( _box$1 ) ? INTERSECTED : NOT_INTERSECTED;

				}

			},
			intersectsPoint: ( point, index ) => {

				const rayPointDistanceSq = _ray$2.distanceSqToPoint( point );

				if ( rayPointDistanceSq < localThresholdSq ) {

					const intersectPoint = new Vector3();

					_ray$2.closestPointToPoint( point, intersectPoint );
					intersectPoint.applyMatrix4( matrixWorld );

					const distance = raycaster.ray.origin.distanceTo( intersectPoint );

					if ( distance < raycaster.near || distance > raycaster.far ) return;

					if ( firstHitOnly && distance >= closestDistance ) return;
					closestDistance = distance;

					index = this.resolvePrimitiveIndex( index );

					closestHit = {
						distance,
						// TODO: this doesn't seem right?
						distanceToRay: Math.sqrt( rayPointDistanceSq ),
						point: intersectPoint,
						index: geometry.index ? geometry.index.getX( index ) : index,
						face: null,
						faceIndex: null,
						barycoord: null,
						object,
					};

					if ( ! firstHitOnly ) {

						intersects.push( closestHit );

					}

				}

			},
		} );

		if ( firstHitOnly && closestHit ) {

			intersects.push( closestHit );

		}

		return intersects;

	}

}

function iterateOverPoints(
	offset,
	count,
	bvh,
	intersectsPointFunc,
	contained,
	depth,
	point
) {

	const { geometry } = bvh;
	const { index } = geometry;
	const pos = geometry.attributes.position;

	for ( let i = offset, l = count + offset; i < l; i ++ ) {

		const prim = bvh.resolvePrimitiveIndex( i );
		const vertexIndex = index ? index.array[ prim ] : prim;
		point.fromBufferAttribute( pos, vertexIndex );

		if ( intersectsPointFunc( point, i, contained, depth ) ) {

			return true;

		}

	}

	return false;

}

/** @import { Object3D } from 'three' */
/** @import { IntersectsBoundsCallback, IntersectsRangeCallback, BoundsTraverseOrderCallback } from './BVH.js' */

const _geometry = /* @__PURE__ */ new BufferGeometry();
const _matrix$1 = /* @__PURE__ */ new Matrix4();
const _inverseMatrix$1 = /* @__PURE__ */ new Matrix4();
const _box = /* @__PURE__ */ new Box3();
const _sphere = /* @__PURE__ */ new Sphere();
const _vec$1 = /* @__PURE__ */ new Vector3();
const _ray$1 = /* @__PURE__ */ new Ray();
const _mesh$1 = /* @__PURE__ */ new Mesh();
const _geometryRange = {};

/**
 * @callback IntersectsObjectCallback
 * @param {Object3D} object - The scene object whose bounds were intersected.
 * @param {number} instanceId - Instance index for InstancedMesh/BatchedMesh, or 0 for regular objects.
 * @param {boolean} contained - Whether the node bounds are fully contained by the query shape.
 * @param {number} depth - The depth of the node in the tree.
 * @returns {boolean} Return `true` to stop traversal.
 */

/**
 * BVH built from a scene hierarchy rather than a single geometry. Each leaf holds
 * one Object3D (or one instance of an InstancedMesh/BatchedMesh), enabling
 * accelerated raycasting and spatial queries across many objects at once.
 *
 * @param {Object3D | Array<Object3D>} root - Root object or array of objects.
 * @param {Object} [options] - Accepts all standard BVH options plus:
 * @param {boolean} [options.precise=false] - Use vertex-level bounds instead of cached bounding boxes.
 * @param {boolean} [options.includeInstances=true] - Treat each instance of InstancedMesh/BatchedMesh as a separate primitive.
 * @extends BVH
 */
class ObjectBVH extends BVH {

	constructor( root, options = {} ) {

		options = {
			precise: false,
			includeInstances: true,
			matrixWorld: Array.isArray( root ) ? new Matrix4() : root.matrixWorld,
			targetLeafSize: 1,
			...options,
		};

		super();

		// collect all the leaf node objects in the geometries
		const objectSet = new Set();
		collectObjects( root, objectSet );

		// calculate the number of bits required for the primary id, leaving the remainder
		// for the instanceId count
		const objects = Array.from( objectSet );
		const idBits = Math.ceil( Math.log2( objects.length ) );
		const idMask = ( 1 << idBits ) - 1;

		this.objects = objects;
		this.idBits = idBits;
		this.idMask = idMask;
		this.primitiveBuffer = null;
		this.primitiveBufferStride = 1;

		// settings
		this.precise = options.precise;
		this.includeInstances = options.includeInstances;
		this.matrixWorld = options.matrixWorld;

		this.init( options );

	}

	/**
	 * Returns the `Object3D` associated with a composite id as provided to `intersectsObject`.
	 * @param {number} compositeId
	 * @returns {Object3D}
	 */
	getObjectFromId( compositeId ) {

		const { idMask, objects } = this;
		const id = getObjectId( compositeId, idMask );
		return objects[ id ];

	}

	/**
	 * Returns the instance index associated with a composite id as provided to `intersectsObject`.
	 * @param {number} compositeId
	 * @returns {number}
	 */
	getInstanceFromId( compositeId ) {

		const { idMask, idBits } = this;
		return getInstanceId( compositeId, idBits, idMask );

	}

	init( options ) {

		const { objects, idBits, matrixWorld } = this;

		// pre-cache the inverse matrix for use in the "getPrimitiveBoundingBox" function
		_inverseMatrix$1.copy( matrixWorld ).invert();

		this.primitiveBuffer = new Uint32Array( this._countPrimitives( objects ) );
		this._fillPrimitiveBuffer( objects, idBits, this.primitiveBuffer );

		super.init( options );

	}

	refit( ...args ) {

		// pre-cache the inverse matrix for use in the "getPrimitiveBoundingBox" function
		_inverseMatrix$1.copy( this.matrixWorld ).invert();

		super.refit( ...args );

	}

	writePrimitiveBounds( i, targetBuffer, writeOffset ) {

		const { primitiveBuffer } = this;
		this._getPrimitiveBoundingBox( primitiveBuffer[ i ], _inverseMatrix$1, _box );

		const { min, max } = _box;
		targetBuffer[ writeOffset + 0 ] = min.x;
		targetBuffer[ writeOffset + 1 ] = min.y;
		targetBuffer[ writeOffset + 2 ] = min.z;
		targetBuffer[ writeOffset + 3 ] = max.x;
		targetBuffer[ writeOffset + 4 ] = max.y;
		targetBuffer[ writeOffset + 5 ] = max.z;

	}

	getRootRanges() {

		return [ { offset: 0, count: this.primitiveBuffer.length } ];

	}

	/**
	 * Performs a spatial query against the BVH. Extends the base `shapecast` with an
	 * `intersectsObject` callback that is called once per object primitive in leaf nodes.
	 *
	 * @param {Object} callbacks
	 * @param {IntersectsBoundsCallback} callbacks.intersectsBounds
	 * @param {IntersectsObjectCallback} [callbacks.intersectsObject]
	 * @param {IntersectsRangeCallback} [callbacks.intersectsRange]
	 * @param {BoundsTraverseOrderCallback} [callbacks.boundsTraverseOrder]
	 * @returns {boolean}
	 */
	shapecast( callbacks ) {

		return super.shapecast( {
			...callbacks,

			intersectsPrimitive: callbacks.intersectsObject,
			scratchPrimitive: null,
			iterate: iterateOverObjects,
		} );

	}

	// TODO: this is out of sync with the MeshBVH raycast signature.
	// Change this to "raycastObject3D"? Or add an equivalent?
	raycast( raycaster, intersects = [] ) {

		const { matrixWorld, includeInstances } = this;
		const { firstHitOnly } = raycaster;
		const localIntersects = [];

		// transform the ray into the local bvh frame
		_inverseMatrix$1.copy( matrixWorld ).invert();
		_ray$1.copy( raycaster.ray ).applyMatrix4( _inverseMatrix$1 );

		let closestDistance = Infinity;
		let closestHit = null;

		this.shapecast( {
			boundsTraverseOrder: box => {

				return box.distanceToPoint( _ray$1.origin );

			},
			intersectsBounds: box => {

				if ( firstHitOnly ) {

					if ( ! _ray$1.intersectBox( box, _vec$1 ) ) {

						return NOT_INTERSECTED;

					}

					let dist;
					if ( box.containsPoint( _ray$1.origin ) ) {

						dist = 0;

					} else {

						_vec$1.applyMatrix4( matrixWorld );
						dist = raycaster.ray.origin.distanceTo( _vec$1 );

					}

					// early out if the box is further than the closest raycast
					return dist < closestDistance ? INTERSECTED : NOT_INTERSECTED;

				} else {

					return _ray$1.intersectsBox( box ) ? INTERSECTED : NOT_INTERSECTED;

				}

			},
			intersectsObject( object, instanceId ) {

				// skip non visible objects
				if ( ! object.visible ) {

					return;

				}

				localIntersects.length = 0;

				if ( object.isInstancedMesh && includeInstances ) {

					// raycast the instance
					_mesh$1.geometry = object.geometry;
					_mesh$1.material = object.material;

					object.getMatrixAt( instanceId, _mesh$1.matrixWorld );
					_mesh$1.matrixWorld.premultiply( object.matrixWorld );
					_mesh$1.raycast( raycaster, localIntersects );

					localIntersects.forEach( hit => {

						hit.object = object;
						hit.instanceId = instanceId;

					} );

					_mesh$1.material = null;

				} else if ( object.isBatchedMesh && includeInstances ) {

					if ( ! object.getVisibleAt( instanceId ) ) {

						return;

					}

					// extract the geometry & material
					const geometryId = object.getGeometryIdAt( instanceId );
					const geometryRange = object.getGeometryRangeAt( geometryId, _geometryRange );

					_geometry.index = object.geometry.index;
					_geometry.attributes = object.geometry.attributes;
					_geometry.setDrawRange( geometryRange.start, geometryRange.count );

					_mesh$1.geometry = _geometry;
					_mesh$1.material = object.material;

					// perform a raycast against the proxy mesh
					object.getMatrixAt( instanceId, _mesh$1.matrixWorld );
					_mesh$1.matrixWorld.premultiply( object.matrixWorld );
					_mesh$1.raycast( raycaster, localIntersects );

					// fix up the fields
					localIntersects.forEach( hit => {

						hit.object = object;
						hit.batchId = instanceId;

					} );

					_mesh$1.material = null;
					_geometry.index = null;
					_geometry.attributes = null;
					_geometry.setDrawRange( 0, Infinity );

				} else {

					object.raycast( raycaster, localIntersects );

				}

				// find the closest hit to track
				if ( firstHitOnly ) {

					localIntersects.forEach( hit => {

						if ( hit.distance < closestDistance ) {

							closestDistance = hit.distance;
							closestHit = hit;

						}

					} );

				} else {

					intersects.push( ...localIntersects );

				}

			},
		} );

		// save the closest hit only if firstHitOnly = true
		if ( firstHitOnly && closestHit ) {

			intersects.push( closestHit );

		}

		return intersects;

	}

	// get the bounding box of a primitive node accounting for the bvh options
	_getPrimitiveBoundingBox( compositeId, inverseMatrixWorld, target ) {

		const { objects, idMask, idBits, precise, includeInstances } = this;
		const id = getObjectId( compositeId, idMask );
		const instanceId = getInstanceId( compositeId, idBits, idMask );
		const object = objects[ id ];

		if ( ! includeInstances && ( object.isInstancedMesh || object.isBatchedMesh ) ) {

			// if we're not using instances then just account for the overall bounds of the BatchedMesh and InstancedMesh
			if ( ! object.boundingBox ) {

				object.computeBoundingBox();

			}

			if ( ! object.boundingSphere ) {

				object.computeBoundingSphere();

			}

			_matrix$1
				.copy( object.matrixWorld )
				.premultiply( inverseMatrixWorld );

			_sphere
				.copy( object.boundingSphere )
				.applyMatrix4( _matrix$1 );

			target
				.copy( object.boundingBox )
				.applyMatrix4( _matrix$1 );

			shrinkToSphere( target, _sphere );

		} else if ( precise ) {

			// calculate precise bounds if necessary by calculating the bounds of all vertices
			// in the bvh frame
			if ( object.isInstancedMesh ) {

				object
					.getMatrixAt( instanceId, _matrix$1 );

				_matrix$1
					.premultiply( object.matrixWorld )
					.premultiply( inverseMatrixWorld );

				getPreciseBounds( object.geometry, _matrix$1, target );

			} else if ( object.isBatchedMesh ) {

				const geometryId = object.getGeometryIdAt( instanceId );
				const geometryRange = object.getGeometryRangeAt( geometryId, _geometryRange );

				_geometry.index = object.geometry.index;
				_geometry.attributes = object.geometry.attributes;
				_geometry.setDrawRange( geometryRange.start, geometryRange.count );

				object
					.getMatrixAt( instanceId, _matrix$1 );

				_matrix$1
					.premultiply( object.matrixWorld )
					.premultiply( inverseMatrixWorld );

				getPreciseBounds( _geometry, _matrix$1, target );

				_geometry.attributes = null;

			} else {

				_matrix$1
					.copy( object.matrixWorld )
					.premultiply( inverseMatrixWorld );

				target.setFromObject( object, true ).applyMatrix4( inverseMatrixWorld );

			}

		} else {

			// otherwise use the fast path of extracting the cached, AABB bounds and transforming them
			// into the local BVH frame
			if ( object.isInstancedMesh ) {

				if ( ! object.geometry.boundingBox ) {

					object.geometry.computeBoundingBox();

				}

				if ( ! object.geometry.boundingSphere ) {

					object.geometry.computeBoundingSphere();

				}

				object
					.getMatrixAt( instanceId, _matrix$1 );

				_matrix$1
					.premultiply( object.matrixWorld )
					.premultiply( inverseMatrixWorld );

				_sphere
					.copy( object.geometry.boundingSphere )
					.applyMatrix4( _matrix$1 );

				target
					.copy( object.geometry.boundingBox )
					.applyMatrix4( _matrix$1 );

				shrinkToSphere( target, _sphere );

			} else if ( object.isBatchedMesh ) {

				const geometryId = object.getGeometryIdAt( instanceId );

				object
					.getMatrixAt( instanceId, _matrix$1 );

				_matrix$1
					.premultiply( object.matrixWorld )
					.premultiply( inverseMatrixWorld );

				object
					.getBoundingSphereAt( geometryId, _sphere )
					.applyMatrix4( _matrix$1 );

				object
					.getBoundingBoxAt( geometryId, target )
					.applyMatrix4( _matrix$1 );

				shrinkToSphere( target, _sphere );

			} else {

				target
					.setFromObject( object, false )
					.applyMatrix4( inverseMatrixWorld );

			}

		}

	}

	// counts the total number of primitives required by the objects in given array of objects
	_countPrimitives( objects ) {

		const { includeInstances } = this;
		let total = 0;
		objects.forEach( object => {

			if ( object.isInstancedMesh && includeInstances ) {

				total += object.count;

			} else if ( object.isBatchedMesh && includeInstances ) {

				if ( ! ( 'instanceCount' in object ) ) {

					throw new Error( 'ObjectBVH: Three.js revision >= r169 is required to use BatchedMesh.' );

				}

				total += object.instanceCount;

			} else {

				total ++;

			}

		} );

		return total;

	}

	_fillPrimitiveBuffer( objects, idBits, target ) {

		const { includeInstances } = this;
		let index = 0;
		objects.forEach( ( object, i ) => {

			if ( object.isInstancedMesh && includeInstances ) {

				const count = object.count;
				for ( let c = 0; c < count; c ++ ) {

					target[ index ] = ( c << idBits ) | i;
					index ++;

				}

			} else if ( object.isBatchedMesh && includeInstances ) {

				const { instanceCount, maxInstanceCount } = object;
				let foundInstances = 0;
				let iter = 0;

				while ( foundInstances < instanceCount && iter < maxInstanceCount ) {

					// TODO: it would be better to have a consistent way of querying whether an
					// instance were active
					try {

						object.getVisibleAt( iter );

						target[ index ] = ( iter << idBits ) | i;
						foundInstances ++;
						index ++;

					} catch {

						//

					}

					iter ++;

				}

			} else {

				target[ index ] = i;
				index ++;

			}

		} );

	}

}

// id functions
// extract the primary object id given the provided mask
function getObjectId( id, idMask ) {

	return id & idMask;

}

// extract the instance id given the mask and number of bits to shift
function getInstanceId( id, idBits, idMask ) {

	return ( id & ( ~ idMask ) ) >>> idBits;

}

// traverse the full scene and collect all leaves
function collectObjects( root, objectSet = new Set() ) {

	if ( Array.isArray( root ) ) {

		root.forEach( object => collectObjects( object, objectSet ) );

	} else {

		root.traverse( child => {

			if ( child.isMesh || child.isLine || child.isPoints ) {

				objectSet.add( child );

			}

		} );

	}

}

// calculate precise box bounds of the given geometry in the given frame
function getPreciseBounds( geometry, matrix, target ) {

	target.makeEmpty();

	const drawRange = geometry.drawRange;
	const indexAttr = geometry.index;
	const posAttr = geometry.attributes.position;
	const start = drawRange.start;
	const vertCount = indexAttr ? indexAttr.count : posAttr.count;
	const count = Math.min( vertCount - start, drawRange.count );
	for ( let i = start, l = start + count; i < l; i ++ ) {

		let vi = i;
		if ( indexAttr ) {

			vi = indexAttr.getX( vi );

		}

		_vec$1.fromBufferAttribute( posAttr, vi ).applyMatrix4( matrix );
		target.expandByPoint( _vec$1 );

	}

	return target;

}

// iterator helper for raycasting
function iterateOverObjects( offset, count, bvh, callback, contained, depth, /* scratch */ ) {

	const { primitiveBuffer, objects, idMask, idBits } = bvh;
	for ( let i = offset, l = count + offset; i < l; i ++ ) {

		const compositeId = primitiveBuffer[ i ];
		const id = getObjectId( compositeId, idMask );
		const instanceId = getInstanceId( compositeId, idBits, idMask );
		const object = objects[ id ];
		if ( callback( object, instanceId, contained, depth ) ) {

			return true;

		}

	}

	return false;

}

function shrinkToSphere( box, sphere ) {

	_vec$1.copy( sphere.center ).addScalar( - sphere.radius );
	box.min.max( _vec$1 );

	_vec$1.copy( sphere.center ).addScalar( sphere.radius );
	box.max.min( _vec$1 );

}

/** @import { SkinnedMesh } from 'three' */
/** @import { IntersectsBoundsCallback, IntersectsRangeCallback, BoundsTraverseOrderCallback } from './BVH.js' */

const _v0 = /* @__PURE__ */ new Vector3();
const _v1 = /* @__PURE__ */ new Vector3();
const _v2 = /* @__PURE__ */ new Vector3();
const _ray = /* @__PURE__ */ new Ray();
const _inverseMatrix = /* @__PURE__ */ new Matrix4();
const _localPoint = /* @__PURE__ */ new Vector3();
const _vec = /* @__PURE__ */ new Vector3();
const _axes = [ 'x', 'y', 'z' ];

const IS_GT_REVISION_169 = parseInt( REVISION ) >= 169;
const IS_LT_REVISION_161 = parseInt( REVISION ) <= 161;

const _uvA = /* @__PURE__ */ new Vector2();
const _uvB = /* @__PURE__ */ new Vector2();
const _uvC = /* @__PURE__ */ new Vector2();
const _normalA = /* @__PURE__ */ new Vector3();
const _normalB = /* @__PURE__ */ new Vector3();
const _normalC = /* @__PURE__ */ new Vector3();

/**
 * @callback IntersectsTriangleCallback
 * @param {ExtendedTriangle} triangle - The triangle primitive in local space.
 * @param {number} index - The primitive index within the BVH buffer.
 * @param {boolean} contained - Whether the node bounds are fully contained by the query shape.
 * @param {number} depth - The depth of the node in the tree.
 * @returns {boolean} Return `true` to stop traversal.
 */

/**
 * BVH for `SkinnedMesh` objects. Computes primitive bounds using
 * `SkinnedMesh.getVertexPosition` so the tree reflects the current posed state
 * of the mesh. Call `refit()` after updating the skeleton to keep bounds accurate.
 *
 * @param {SkinnedMesh} mesh
 * @param {Object} [options] - Same options as {@link GeometryBVH}.
 * @extends GeometryBVH
 */
class SkinnedMeshBVH extends GeometryBVH {

	get primitiveStride() {

		return 3;

	}

	constructor( mesh, options = {} ) {

		if ( ! mesh.isMesh ) {

			throw new Error( 'SkinnedMeshBVH: First argument must be a Mesh.' );

		}

		// skip generation initially so we can add our local fields
		// TODO: is there a more clean way to handle this? Update all subclasses to be
		// responsible for calling "init" themselves?
		super( mesh.geometry, {
			...options,
			[ SKIP_GENERATION ]: true,
		} );
		this.mesh = mesh;

		if ( ! options[ SKIP_GENERATION ] ) {

			this.init( options );

		}

	}

	writePrimitiveBounds( i, targetBuffer, baseIndex ) {

		const { mesh, geometry } = this;
		const indirectBuffer = this._indirectBuffer;
		const index = geometry.index ? geometry.index.array : null;

		const tri = indirectBuffer ? indirectBuffer[ i ] : i;
		const tri3 = tri * 3;

		let ai = tri3 + 0;
		let bi = tri3 + 1;
		let ci = tri3 + 2;

		if ( index ) {

			ai = index[ ai ];
			bi = index[ bi ];
			ci = index[ ci ];

		}

		// Get skinned vertex positions
		mesh.getVertexPosition( ai, _v0 );
		mesh.getVertexPosition( bi, _v1 );
		mesh.getVertexPosition( ci, _v2 );

		// Compute bounds for each axis
		for ( let el = 0; el < 3; el ++ ) {

			const axis = _axes[ el ];
			const a = _v0[ axis ];
			const b = _v1[ axis ];
			const c = _v2[ axis ];

			let min = a;
			if ( b < min ) min = b;
			if ( c < min ) min = c;

			let max = a;
			if ( b > max ) max = b;
			if ( c > max ) max = c;

			// Write in min/max format [minx, miny, minz, maxx, maxy, maxz]
			targetBuffer[ baseIndex + el ] = min;
			targetBuffer[ baseIndex + el + 3 ] = max;

		}

		return targetBuffer;

	}

	/**
	 * Performs a spatial query against the BVH. Extends the base `shapecast` with an
	 * `intersectsTriangle` callback that is called once per triangle primitive in leaf nodes.
	 *
	 * @param {Object} callbacks
	 * @param {IntersectsBoundsCallback} callbacks.intersectsBounds
	 * @param {IntersectsTriangleCallback} [callbacks.intersectsTriangle]
	 * @param {IntersectsRangeCallback} [callbacks.intersectsRange]
	 * @param {BoundsTraverseOrderCallback} [callbacks.boundsTraverseOrder]
	 * @returns {boolean}
	 */
	shapecast( callbacks ) {

		const triangle = new ExtendedTriangle();
		return super.shapecast(
			{
				...callbacks,
				intersectsPrimitive: callbacks.intersectsTriangle,
				scratchPrimitive: triangle,
				iterate: iterateOverTriangles,
			},
		);

	}

	raycastObject3D( object, raycaster, intersects = [] ) {

		const { material } = object;
		if ( material === undefined ) {

			return;

		}

		const { matrixWorld } = object;
		const { firstHitOnly } = raycaster;

		_inverseMatrix.copy( matrixWorld ).invert();
		_ray.copy( raycaster.ray ).applyMatrix4( _inverseMatrix );

		let closestHit = null;
		let closestDistance = Infinity;

		this.shapecast( {
			boundsTraverseOrder: box => {

				return box.distanceToPoint( _ray.origin );

			},
			intersectsBounds: box => {

				if ( firstHitOnly ) {

					if ( ! _ray.intersectBox( box, _vec ) ) {

						return NOT_INTERSECTED;

					}

					let dist;
					if ( box.containsPoint( _ray.origin ) ) {

						dist = 0;

					} else {

						_vec.applyMatrix4( matrixWorld );
						dist = raycaster.ray.origin.distanceTo( _vec );

					}

					// early out if the box is further than the closest raycast
					return dist < closestDistance ? INTERSECTED : NOT_INTERSECTED;

				} else {

					return _ray.intersectsBox( box ) ? INTERSECTED : NOT_INTERSECTED;

				}

			},
			intersectsTriangle: ( tri, triIndex ) => {

				// get the intersection
				let point = null;
				if ( material.side === FrontSide ) {

					point = _ray.intersectTriangle( tri.a, tri.b, tri.c, true, _localPoint );

				} else if ( material.side === BackSide ) {

					point = _ray.intersectTriangle( tri.c, tri.b, tri.a, true, _localPoint );

				} else {

					point = _ray.intersectTriangle( tri.a, tri.b, tri.c, false, _localPoint );

				}

				if ( ! point ) {

					return;

				}

				// transform it into world space
				point = point.clone().applyMatrix4( matrixWorld );

				// check distance to ray
				const dist = raycaster.ray.origin.distanceTo( point );
				if ( dist >= raycaster.near && dist <= raycaster.far ) {

					if ( firstHitOnly && dist >= closestDistance ) {

						return;

					}

					// get the vertex indices
					const { geometry } = this;
					const { index } = geometry;
					const actualTri = this.resolvePrimitiveIndex( triIndex );
					const triOffset = actualTri * 3;

					let ai = triOffset + 0;
					let bi = triOffset + 1;
					let ci = triOffset + 2;

					if ( index ) {

						ai = index.array[ ai ];
						bi = index.array[ bi ];
						ci = index.array[ ci ];

					}

					// build the intersection result
					const hit = {
						distance: dist,
						point: point.clone(),
						object,
						uv: null,
						uv1: null,
						normal: null,
						face: {
							a: ai,
							b: bi,
							c: ci,
							normal: Triangle.getNormal( tri.a, tri.b, tri.c, new Vector3() ),
							materialIndex: 0
						},
						faceIndex: actualTri,
					};

					if ( IS_GT_REVISION_169 ) {

						const barycoord = new Vector3();
						Triangle.getBarycoord( _localPoint, tri.a, tri.b, tri.c, barycoord );
						hit.barycoord = barycoord;

					}

					// add attribute fields if available
					const uv = geometry.attributes.uv;
					const uv1 = geometry.attributes.uv1;
					const normal = geometry.attributes.normal;

					if ( uv ) {

						_uvA.fromBufferAttribute( uv, ai );
						_uvB.fromBufferAttribute( uv, bi );
						_uvC.fromBufferAttribute( uv, ci );

						hit.uv = new Vector2();
						const resUv = Triangle.getInterpolation( _localPoint, tri.a, tri.b, tri.c, _uvA, _uvB, _uvC, hit.uv );
						if ( ! IS_GT_REVISION_169 ) hit.uv = resUv;

					}

					if ( uv1 ) {

						_uvA.fromBufferAttribute( uv1, ai );
						_uvB.fromBufferAttribute( uv1, bi );
						_uvC.fromBufferAttribute( uv1, ci );

						hit.uv1 = new Vector2();
						const resUv1 = Triangle.getInterpolation( _localPoint, tri.a, tri.b, tri.c, _uvA, _uvB, _uvC, hit.uv1 );
						if ( ! IS_GT_REVISION_169 ) hit.uv1 = resUv1;
						if ( IS_LT_REVISION_161 ) hit.uv2 = hit.uv1;

					}

					if ( normal ) {

						_normalA.fromBufferAttribute( normal, ai );
						_normalB.fromBufferAttribute( normal, bi );
						_normalC.fromBufferAttribute( normal, ci );

						hit.normal = new Vector3();
						const resNormal = Triangle.getInterpolation( _localPoint, tri.a, tri.b, tri.c, _normalA, _normalB, _normalC, hit.normal );
						if ( hit.normal.dot( _ray.direction ) > 0 ) {

							hit.normal.multiplyScalar( - 1 );

						}

						if ( ! IS_GT_REVISION_169 ) hit.normal = resNormal;

					}

					// first hit only settings
					closestDistance = hit.distance;
					closestHit = hit;

					if ( ! firstHitOnly ) {

						intersects.push( hit );

					}

				}

			}
		} );

		if ( firstHitOnly && closestHit ) {

			intersects.push( closestHit );

		}

		return intersects;

	}

}

function iterateOverTriangles(
	offset,
	count,
	bvh,
	intersectsTriangleFunc,
	contained,
	depth,
	triangle
) {

	const { mesh, geometry } = bvh;
	const index = geometry.index ? geometry.index.array : null;

	for ( let i = offset, l = count + offset; i < l; i ++ ) {

		const tri = bvh.resolvePrimitiveIndex( i );

		let i0 = 3 * tri + 0;
		let i1 = 3 * tri + 1;
		let i2 = 3 * tri + 2;

		if ( index ) {

			i0 = index[ i0 ];
			i1 = index[ i1 ];
			i2 = index[ i2 ];

		}

		mesh.getVertexPosition( i0, triangle.a );
		mesh.getVertexPosition( i1, triangle.b );
		mesh.getVertexPosition( i2, triangle.c );
		triangle.needsUpdate = true;

		if ( intersectsTriangleFunc( triangle, i, contained, depth ) ) {

			return true;

		}

	}

	return false;

}

/** @import { Color } from 'three' */
/** @import { GeometryBVH } from '../core/GeometryBVH.js' */

const boundingBox = /* @__PURE__ */ new Box3();
const matrix = /* @__PURE__ */ new Matrix4();
const vec = /* @__PURE__ */ new Vector3();

class BVHRootHelper extends Object3D {

	get isMesh() {

		return ! this.displayEdges;

	}

	get isLineSegments() {

		return this.displayEdges;

	}

	get isLine() {

		return this.displayEdges;

	}

	getVertexPosition( ...args ) {

		// implement this function so it works with Box3.setFromObject
		return Mesh.prototype.getVertexPosition.call( this, ...args );

	}

	constructor( bvh, material, depth = 10, group = 0 ) {

		super();

		this.material = material;
		this.geometry = new BufferGeometry();
		this.name = 'BVHRootHelper';
		this.depth = depth;
		this.displayParents = false;
		this.bvh = bvh;
		this.displayEdges = true;
		this._group = group;

	}

	raycast() {}

	update() {

		const boundsTree = this.bvh;
		this.geometry.dispose();
		this.visible = false;
		if ( boundsTree ) {

			this.geometry = this.getGeometry( boundsTree );
			this.visible = true;

		}

	}

	getGeometry( boundsTree ) {

		const group = this._group;

		// fill in the position buffer with the bounds corners
		let positionArray = null;
		if ( group !== - 1 ) {

			positionArray = this.getBVHBoundPositions( boundsTree, group );

		} else {

			const positionArrays = boundsTree._roots.map( ( r, i ) => this.getBVHBoundPositions( boundsTree, i ) );
			const total = positionArrays.reduce( ( v, arr ) => v + arr.length, 0 );
			positionArray = new Float32Array( total );

			let offset = 0;
			positionArrays.forEach( arr => {

				positionArray.set( arr, offset );
				offset += arr.length;

			} );

		}

		const indexArray = this.getBVHBoundIndices( positionArray );

		// update the geometry
		const geometry = new BufferGeometry();
		geometry.setIndex( new BufferAttribute( indexArray, 1, false ) );
		geometry.setAttribute( 'position', new BufferAttribute( positionArray, 3, false ) );
		return geometry;

	}

	getBVHBoundIndices( positionArray ) {

		const boundsCount = positionArray.length / ( 8 * 3 );
		let indexArray;
		let indices;
		if ( this.displayEdges ) {

			// fill in the index buffer to point to the corner points
			indices = new Uint8Array( [
				// x axis
				0, 4,
				1, 5,
				2, 6,
				3, 7,

				// y axis
				0, 2,
				1, 3,
				4, 6,
				5, 7,

				// z axis
				0, 1,
				2, 3,
				4, 5,
				6, 7,
			] );

		} else {

			indices = new Uint8Array( [

				// X-, X+
				0, 1, 2,
				2, 1, 3,

				4, 6, 5,
				6, 7, 5,

				// Y-, Y+
				1, 4, 5,
				0, 4, 1,

				2, 3, 6,
				3, 7, 6,

				// Z-, Z+
				0, 2, 4,
				2, 6, 4,

				1, 5, 3,
				3, 5, 7,

			] );

		}

		if ( positionArray.length > 65535 ) {

			indexArray = new Uint32Array( indices.length * boundsCount );

		} else {

			indexArray = new Uint16Array( indices.length * boundsCount );

		}

		const indexLength = indices.length;
		for ( let i = 0; i < boundsCount; i ++ ) {

			const posOffset = i * 8;
			const indexOffset = i * indexLength;
			for ( let j = 0; j < indexLength; j ++ ) {

				indexArray[ indexOffset + j ] = posOffset + indices[ j ];

			}

		}

		return indexArray;

	}

	getBVHBoundPositions( bvh, group = 0, matrix = null ) {

		// count the number of bounds required
		const targetDepth = this.depth - 1;
		const displayParents = this.displayParents;
		let boundsCount = 0;
		bvh.traverse( ( depth, isLeaf ) => {

			if ( depth >= targetDepth || isLeaf ) {

				boundsCount ++;
				return true;

			} else if ( displayParents ) {

				boundsCount ++;

			}

		}, group );

		// fill in the position buffer with the bounds corners
		let posIndex = 0;
		const positionArray = new Float32Array( 8 * 3 * boundsCount );
		bvh.traverse( ( depth, isLeaf, boundingData ) => {

			const terminate = depth >= targetDepth || isLeaf;
			if ( terminate || displayParents ) {

				arrayToBox( 0, boundingData, boundingBox );

				const { min, max } = boundingBox;
				for ( let x = - 1; x <= 1; x += 2 ) {

					const xVal = x < 0 ? min.x : max.x;
					for ( let y = - 1; y <= 1; y += 2 ) {

						const yVal = y < 0 ? min.y : max.y;
						for ( let z = - 1; z <= 1; z += 2 ) {

							const zVal = z < 0 ? min.z : max.z;
							vec.set( xVal, yVal, zVal );
							if ( matrix ) {

								vec.applyMatrix4( matrix );

							}

							vec.toArray( positionArray, posIndex );
							posIndex += 3;

						}

					}

				}

				return terminate;

			}

		}, group );

		return positionArray;

	}

}

/**
 * A `THREE.Group` that visualizes a BVH as wireframe bounding boxes or solid
 * face overlays. Attach it as a sibling of the mesh in the scene graph and
 * call `update()` whenever the mesh's BVH or world transform changes.
 *
 * @param {Object3D | GeometryBVH | null} [mesh=null] - The mesh whose `geometry.boundsTree`
 *   should be displayed, or a `GeometryBVH` to display directly.
 * @param {GeometryBVH | number | null} [bvh=null] - The BVH to visualize. When the first
 *   argument is a `GeometryBVH`, this argument is interpreted as `depth`.
 * @param {number} [depth=10] - Maximum tree depth to display.
 * @extends Group
 */
class BVHHelper extends Group {

	/**
	 * Shortcut to `edgeMaterial.color`.
	 * @type {Color}
	 * @readonly
	 */
	get color() {

		return this.edgeMaterial.color;

	}

	/**
	 * Opacity applied to both edge and mesh materials.
	 * @type {number}
	 */
	get opacity() {

		return this.edgeMaterial.opacity;

	}

	set opacity( v ) {

		this.edgeMaterial.opacity = v;
		this.meshMaterial.opacity = v;

	}

	get objectIndex() {

		console.warn( 'BVHHelper: "objectIndex" has been renamed "instanceId".' );
		return this.instanceId;

	}

	set objectIndex( v ) {

		console.warn( 'BVHHelper: "objectIndex" has been renamed "instanceId".' );
		this.instanceId = v;

	}

	constructor( mesh = null, bvh = null, depth = 10 ) {

		// handle bvh, depth signature
		if ( mesh instanceof MeshBVH ) {

			depth = bvh || 10;
			bvh = mesh;
			mesh = null;

		}

		// handle mesh, depth signature
		if ( typeof bvh === 'number' ) {

			depth = bvh;
			bvh = null;

		}

		super();

		this.name = 'BVHHelper';

		/** @type {number} */
		this.depth = depth;

		/** @type {Object3D | null} */
		this.mesh = mesh;

		/** @type {GeometryBVH | null} */
		this.bvh = bvh;

		/** @type {boolean} */
		this.displayParents = false;

		/** @type {boolean} */
		this.displayEdges = true;

		/** @type {number} */
		this.instanceId = 0;

		this._roots = [];

		const edgeMaterial = new LineBasicMaterial( {
			color: 0x00FF88,
			transparent: true,
			opacity: 0.3,
			depthWrite: false,
		} );

		const meshMaterial = new MeshBasicMaterial( {
			color: 0x00FF88,
			transparent: true,
			opacity: 0.3,
			depthWrite: false,
		} );

		meshMaterial.color = edgeMaterial.color;

		/**
		 * Material used when rendering in wireframe edge mode.
		 * @type {LineBasicMaterial}
		 */
		this.edgeMaterial = edgeMaterial;

		/**
		 * Material used when rendering in solid face mode.
		 * @type {MeshBasicMaterial}
		 */
		this.meshMaterial = meshMaterial;

		this.update();

	}

	/**
	 * Rebuilds the helper's display geometry from the current BVH state. Must
	 * be called after changes to the BVH, `depth`, `displayParents`, or
	 * `displayEdges`.
	 */
	update() {

		const mesh = this.mesh;
		const instanceId = this.instanceId;
		let bvh = this.bvh || mesh.boundsTree || mesh.geometry && mesh.geometry.boundsTree || null;
		if ( mesh && mesh.isBatchedMesh && mesh.boundsTrees && ! bvh && instanceId >= 0 ) {

			// get the bvh from a batchedMesh if not provided
			// TODO: we should have an official way to get the geometry index cleanly
			const drawInfo = mesh._drawInfo[ instanceId ];
			if ( drawInfo ) {

				bvh = mesh.boundsTrees[ drawInfo.geometryIndex ] || bvh;

			}

		}

		const totalRoots = bvh ? bvh._roots.length : 0;
		while ( this._roots.length > totalRoots ) {

			const root = this._roots.pop();
			root.geometry.dispose();
			this.remove( root );

		}

		for ( let i = 0; i < totalRoots; i ++ ) {

			const { depth, edgeMaterial, meshMaterial, displayParents, displayEdges } = this;

			if ( i >= this._roots.length ) {

				const root = new BVHRootHelper( bvh, edgeMaterial, depth, i );
				this.add( root );
				this._roots.push( root );

			}

			const root = this._roots[ i ];
			root.bvh = bvh;
			root.depth = depth;
			root.displayParents = displayParents;
			root.displayEdges = displayEdges;
			root.material = displayEdges ? edgeMaterial : meshMaterial;
			root.update();

		}

	}

	updateMatrixWorld( ...args ) {

		const mesh = this.mesh;
		const parent = this.parent;
		const instanceId = this.instanceId;

		if ( mesh !== null ) {

			mesh.updateWorldMatrix( true, false );

			if ( parent ) {

				this.matrix
					.copy( parent.matrixWorld )
					.invert()
					.multiply( mesh.matrixWorld );

			} else {

				this.matrix
					.copy( mesh.matrixWorld );

			}

			// handle batched and instanced mesh bvhs
			if ( ( mesh.isInstancedMesh || mesh.isBatchedMesh ) && instanceId >= 0 ) {

				mesh.getMatrixAt( instanceId, matrix );
				this.matrix.multiply( matrix );

			}

			this.matrix.decompose(
				this.position,
				this.quaternion,
				this.scale,
			);

		}

		super.updateMatrixWorld( ...args );

	}

	copy( source ) {

		this.depth = source.depth;
		this.mesh = source.mesh;
		this.bvh = source.bvh;
		this.opacity = source.opacity;
		this.color.copy( source.color );

	}

	clone() {

		return new BVHHelper().copy( this );

	}

	/**
	 * Disposes of the materials and geometries used by the helper.
	 */
	dispose() {

		this.edgeMaterial.dispose();
		this.meshMaterial.dispose();

		const children = this.children;
		for ( let i = 0, l = children.length; i < l; i ++ ) {

			children[ i ].geometry.dispose();

		}

	}

}

class MeshBVHHelper extends BVHHelper {

	constructor( ...args ) {

		console.warn( 'MeshBVHHelper: Class has been deprecated. Use BVHHelper instead.' );
		super( ...args );

	}

}

/** @import { MeshBVH } from '../core/MeshBVH.js' */
/** @import { BVH } from '../core/BVH.js' */

const _box1 = /* @__PURE__ */ new Box3();
const _box2 = /* @__PURE__ */ new Box3();

// https://stackoverflow.com/questions/1248302/how-to-get-the-size-of-a-javascript-object
function getElementSize( el ) {

	switch ( typeof el ) {

		case 'number':
			return 8;
		case 'string':
			return el.length * 2;
		case 'boolean':
			return 4;
		default:
			return 0;

	}

}

function isTypedArray( arr ) {

	const regex = /(Uint|Int|Float)(8|16|32)Array/;
	return regex.test( arr.constructor.name );

}

function getRootExtremes( bvh, group ) {

	const result = {
		nodeCount: 0,
		leafNodeCount: 0,

		depth: {
			min: Infinity, max: - Infinity
		},
		primitives: {
			min: Infinity, max: - Infinity
		},
		splits: [ 0, 0, 0 ],
		surfaceAreaScore: 0,
	};

	bvh.traverse( ( depth, isLeaf, boundingData, offsetOrSplit, count ) => {

		const l0 = boundingData[ 0 + 3 ] - boundingData[ 0 ];
		const l1 = boundingData[ 1 + 3 ] - boundingData[ 1 ];
		const l2 = boundingData[ 2 + 3 ] - boundingData[ 2 ];

		const surfaceArea = 2 * ( l0 * l1 + l1 * l2 + l2 * l0 );

		result.nodeCount ++;
		if ( isLeaf ) {

			result.leafNodeCount ++;

			result.depth.min = Math.min( depth, result.depth.min );
			result.depth.max = Math.max( depth, result.depth.max );

			result.primitives.min = Math.min( count, result.primitives.min );
			result.primitives.max = Math.max( count, result.primitives.max );

			result.surfaceAreaScore += surfaceArea * PRIMITIVE_INTERSECT_COST * count;

		} else {

			result.splits[ offsetOrSplit ] ++;

			result.surfaceAreaScore += surfaceArea * TRAVERSAL_COST;

		}

	}, group );

	// If there are no leaf nodes because the tree hasn't finished generating yet.
	if ( result.primitives.min === Infinity ) {

		result.primitives.min = 0;
		result.primitives.max = 0;

	}

	if ( result.depth.min === Infinity ) {

		result.depth.min = 0;
		result.depth.max = 0;

	}

	return result;

}

/**
 * @section Debug Functions
 * @typedef {Object} BVHExtremes
 * @property {number} nodeCount Total number of nodes in the tree including leaf nodes.
 * @property {number} leafNodeCount Total number of leaf nodes in the tree.
 * @property {number} surfaceAreaScore Total tree score based on the surface area heuristic.
 * Lower is better. Useful for comparing tree quality and performance, and for detecting
 * degradation after `MeshBVH.refit` calls.
 * @property {{ min: number, max: number }} depth Min and max depth of leaf nodes.
 * @property {{ min: number, max: number }} tris Min and max triangle count in leaf nodes.
 * @property {Array<number>} splits Number of splits on each axis as a three-element array `[X, Y, Z]`.
 */

/**
 * Measures the min and max extremes of the BVH tree structure, including node
 * depth, leaf primitive count, split axis distribution, and a surface-area
 * heuristic score. Returns one entry per root group in the BVH.
 * @section Debug Functions
 * @param {MeshBVH} bvh
 * @returns {Array<BVHExtremes>}
 */
function getBVHExtremes( bvh ) {

	return bvh._roots.map( ( root, i ) => getRootExtremes( bvh, i ) );

}

/**
 * Roughly estimates the amount of memory in bytes used by a BVH by walking
 * its object graph and summing typed-array byte lengths and primitive sizes.
 * @section Debug Functions
 * @param {BVH} bvh
 * @returns {number}
 */
function estimateMemoryInBytes( obj ) {

	const traversed = new Set();
	const stack = [ obj ];
	let bytes = 0;

	while ( stack.length ) {

		const curr = stack.pop();
		if ( traversed.has( curr ) ) {

			continue;

		}

		traversed.add( curr );

		for ( let key in curr ) {

			if ( ! Object.hasOwn( curr, key ) ) {

				continue;

			}

			bytes += getElementSize( key );

			const value = curr[ key ];
			if ( value && ( typeof value === 'object' || typeof value === 'function' ) ) {

				if ( isTypedArray( value ) ) {

					bytes += value.byteLength;

				} else if ( isSharedArrayBufferSupported() && value instanceof SharedArrayBuffer ) {

					bytes += value.byteLength;

				} else if ( value instanceof ArrayBuffer ) {

					bytes += value.byteLength;

				} else {

					stack.push( value );

				}

			} else {

				bytes += getElementSize( value );

			}


		}

	}

	return bytes;

}

/**
 * Validates that every node's bounding box fully contains its children and,
 * for leaf nodes, fully contains all of its primitives. Uses `console.assert`
 * to log failures and returns `false` if any check fails.
 * @section Debug Functions
 * @param {MeshBVH} bvh
 * @returns {boolean}
 */
function validateBounds( bvh ) {

	const depthStack = [];
	const tempBuffer = new Float32Array( 6 );
	let passes = true;

	bvh.traverse( ( depth, isLeaf, boundingData, offset, count ) => {

		const info = {
			depth,
			isLeaf,
			boundingData,
			offset,
			count,
		};
		depthStack[ depth ] = info;

		arrayToBox( 0, boundingData, _box1 );
		const parent = depthStack[ depth - 1 ];

		if ( isLeaf ) {

			// Compute the actual bounds of the primitives in this leaf
			bvh.writePrimitiveRangeBounds( offset, count, tempBuffer, 0 );

			// tempBuffer is in min/max format [minx, miny, minz, maxx, maxy, maxz]
			_box2.min.set( tempBuffer[ 0 ], tempBuffer[ 1 ], tempBuffer[ 2 ] );
			_box2.max.set( tempBuffer[ 3 ], tempBuffer[ 4 ], tempBuffer[ 5 ] );

			// Check if the stored bounds contain the actual primitive bounds
			const isContained = _box1.containsBox( _box2 );
			console.assert( isContained, 'Leaf bounds does not fully contain primitives.' );
			passes = passes && isContained;

		}

		if ( parent ) {

			// check if my bounds fit in my parents
			arrayToBox( 0, parent.boundingData, _box2 );

			const isContained = _box2.containsBox( _box1 );
			console.assert( isContained, 'Parent bounds does not fully contain child.' );
			passes = passes && isContained;

		}

	} );

	return passes;

}

/**
 * Returns a plain-object tree that mirrors the BVH hierarchy, useful for
 * inspecting or serialising the structure for debugging. Each node has a
 * `bounds` (`Box3`) and either `{ count, offset }` (leaf) or `{ left, right }`
 * (internal) fields.
 * @section Debug Functions
 * @param {BVH} bvh
 * @returns {Object}
 */
function getJSONStructure( bvh ) {

	const depthStack = [];

	bvh.traverse( ( depth, isLeaf, boundingData, offset, count ) => {

		const info = {
			bounds: arrayToBox( 0, boundingData, new Box3() ),
		};

		if ( isLeaf ) {

			info.count = count;
			info.offset = offset;

		} else {

			info.left = null;
			info.right = null;

		}

		depthStack[ depth ] = info;

		// traversal hits the left then right node
		const parent = depthStack[ depth - 1 ];
		if ( parent ) {

			if ( parent.left === null ) {

				parent.left = info;

			} else {

				parent.right = info;

			}

		}

	} );

	return depthStack[ 0 ];

}

/** @import { Raycaster, Intersection } from 'three' */
/** @import { GeometryBVH } from '../core/GeometryBVH.js' */

const IS_REVISION_166 = parseInt( REVISION ) >= 166;

// TODO: how can we expand these raycast functions?
const _raycastFunctions = {
	'Mesh': Mesh.prototype.raycast,
	'Line': Line.prototype.raycast,
	'LineSegments': LineSegments.prototype.raycast,
	'LineLoop': LineLoop.prototype.raycast,
	'Points': Points.prototype.raycast,
	'BatchedMesh': BatchedMesh.prototype.raycast,
};

const _mesh = /* @__PURE__ */ new Mesh();
const _batchIntersects = [];

/**
 * An accelerated raycast function with the same signature as `THREE.Mesh.raycast`. Uses the BVH
 * for raycasting if it's available otherwise it falls back to the built-in approach. The results
 * of the function are designed to be identical to the results of the conventional
 * `THREE.Mesh.raycast` results.
 *
 * If the raycaster object being used has a property `firstHitOnly` set to `true`, then the
 * raycasting will terminate as soon as it finds the closest intersection to the ray's origin and
 * return only that intersection. This is typically several times faster than searching for all
 * intersections.
 *
 * @section Extension Utilities
 * @param {Raycaster} raycaster
 * @param {Array<Intersection>} intersects
 * @returns {void}
 */
function acceleratedRaycast( raycaster, intersects ) {

	if ( this.isBatchedMesh ) {

		acceleratedBatchedMeshRaycast.call( this, raycaster, intersects );

	} else {

		const { geometry } = this;
		if ( geometry.boundsTree ) {

			geometry.boundsTree.raycastObject3D( this, raycaster, intersects );

		} else {

			let raycastFunction;
			if ( this instanceof Mesh ) {

				raycastFunction = _raycastFunctions.Mesh;

			} else if ( this instanceof LineSegments ) {

				raycastFunction = _raycastFunctions.LineSegments;

			} else if ( this instanceof LineLoop ) {

				raycastFunction = _raycastFunctions.LineLoop;

			} else if ( this instanceof Line ) {

				raycastFunction = _raycastFunctions.Line;

			} else if ( this instanceof Points ) {

				raycastFunction = _raycastFunctions.Points;

			} else {

				throw new Error( 'BVH: Fallback raycast function not found.' );

			}

			raycastFunction.call( this, raycaster, intersects );

		}

	}

}

function acceleratedBatchedMeshRaycast( raycaster, intersects ) {

	if ( this.boundsTrees ) {

		// TODO: remove use of geometry info, instance info when r170 is minimum version
		const boundsTrees = this.boundsTrees;
		const drawInfo = this._drawInfo || this._instanceInfo;
		const drawRanges = this._drawRanges || this._geometryInfo;
		const matrixWorld = this.matrixWorld;

		_mesh.material = this.material;
		_mesh.geometry = this.geometry;

		const oldBoundsTree = _mesh.geometry.boundsTree;
		const oldDrawRange = _mesh.geometry.drawRange;

		if ( _mesh.geometry.boundingSphere === null ) {

			_mesh.geometry.boundingSphere = new Sphere();

		}

		// TODO: provide new method to get instances count instead of 'drawInfo.length'
		for ( let i = 0, l = drawInfo.length; i < l; i ++ ) {

			if ( ! this.getVisibleAt( i ) ) {

				continue;

			}

			// TODO: use getGeometryIndex
			const geometryId = drawInfo[ i ].geometryIndex;

			_mesh.geometry.boundsTree = boundsTrees[ geometryId ];

			this.getMatrixAt( i, _mesh.matrixWorld ).premultiply( matrixWorld );

			if ( ! _mesh.geometry.boundsTree ) {

				this.getBoundingBoxAt( geometryId, _mesh.geometry.boundingBox );
				this.getBoundingSphereAt( geometryId, _mesh.geometry.boundingSphere );

				const drawRange = drawRanges[ geometryId ];
				_mesh.geometry.setDrawRange( drawRange.start, drawRange.count );

			}

			_mesh.raycast( raycaster, _batchIntersects );

			for ( let j = 0, l = _batchIntersects.length; j < l; j ++ ) {

				const intersect = _batchIntersects[ j ];
				intersect.object = this;
				intersect.batchId = i;
				intersects.push( intersect );

			}

			_batchIntersects.length = 0;

		}

		_mesh.geometry.boundsTree = oldBoundsTree;
		_mesh.geometry.drawRange = oldDrawRange;
		_mesh.material = null;
		_mesh.geometry = null;

	} else {

		_raycastFunctions.BatchedMesh.call( this, raycaster, intersects );

	}

}

/**
 * A pre-made BufferGeometry extension function that builds a new BVH, assigns it to `boundsTree`
 * for BufferGeometry, and applies the new index buffer to the geometry. Comparable to
 * `computeBoundingBox` and `computeBoundingSphere`.
 *
 * ```js
 * THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
 * ```
 *
 * @section Extension Utilities
 * @param {Object} [options]
 * @returns {GeometryBVH}
 */
function computeBoundsTree( options = {} ) {

	const { type = MeshBVH } = options;
	this.boundsTree = new type( this, options );
	return this.boundsTree;

}

/**
 * A BufferGeometry extension function that disposes of the BVH.
 *
 * ```js
 * THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
 * ```
 *
 * @section Extension Utilities
 * @returns {void}
 */
function disposeBoundsTree() {

	this.boundsTree = null;

}

/**
 * Equivalent of `computeBoundsTree` for `BatchedMesh`. Creates the
 * `BatchedMesh.boundsTrees` array if it does not exist. If `index` is `-1`
 * BVHs for all available geometries are generated and the full array is
 * returned; otherwise only the BVH at that geometry index is generated and
 * returned.
 *
 * ```js
 * THREE.BatchedMesh.prototype.computeBoundsTree = computeBatchedBoundsTree;
 * ```
 *
 * @section Extension Utilities
 * @param {number} [index=-1]
 * @param {Object} [options]
 * @returns {GeometryBVH | Array<GeometryBVH> | null}
 */
function computeBatchedBoundsTree( index = - 1, options = {} ) {

	if ( ! IS_REVISION_166 ) {

		throw new Error( 'BatchedMesh: Three r166+ is required to compute bounds trees.' );

	}

	options = {
		...options,
		range: null
	};

	const drawRanges = this._drawRanges || this._geometryInfo;
	const geometryCount = this._geometryCount;
	if ( ! this.boundsTrees ) {

		this.boundsTrees = new Array( geometryCount ).fill( null );

	}

	const boundsTrees = this.boundsTrees;
	while ( boundsTrees.length < geometryCount ) {

		boundsTrees.push( null );

	}

	if ( index < 0 ) {

		for ( let i = 0; i < geometryCount; i ++ ) {

			options.range = drawRanges[ i ];
			boundsTrees[ i ] = new MeshBVH( this.geometry, options );

		}

		return boundsTrees;

	} else {

		if ( index < drawRanges.length ) {

			options.range = drawRanges[ index ];
			boundsTrees[ index ] = new MeshBVH( this.geometry, options );

		}

		return boundsTrees[ index ] || null;

	}

}

/**
 * Equivalent of `disposeBoundsTree` for `BatchedMesh`. Sets entries in
 * `BatchedMesh.boundsTrees` to `null`. If `index` is `-1` all BVHs are
 * disposed; otherwise only the BVH at that geometry index is disposed.
 *
 * ```js
 * THREE.BatchedMesh.prototype.disposeBoundsTree = disposeBatchedBoundsTree;
 * ```
 *
 * @section Extension Utilities
 * @param {number} [index=-1]
 * @returns {void}
 */
function disposeBatchedBoundsTree( index = - 1 ) {

	if ( index < 0 ) {

		this.boundsTrees.fill( null );

	} else {

		if ( index < this.boundsTrees.length ) {

			this.boundsTrees[ index ] = null;

		}

	}

}

/** @import { BufferAttribute } from 'three' */

function countToStringFormat( count ) {

	switch ( count ) {

		case 1: return 'R';
		case 2: return 'RG';
		case 3: return 'RGBA';
		case 4: return 'RGBA';

	}

	throw new Error();

}

function countToFormat( count ) {

	switch ( count ) {

		case 1: return RedFormat;
		case 2: return RGFormat;
		case 3: return RGBAFormat;
		case 4: return RGBAFormat;

	}

}

function countToIntFormat( count ) {

	switch ( count ) {

		case 1: return RedIntegerFormat;
		case 2: return RGIntegerFormat;
		case 3: return RGBAIntegerFormat;
		case 4: return RGBAIntegerFormat;

	}

}

/**
 * Float, Uint, and Int VertexAttributeTexture implementations are designed to simplify the
 * efficient packing of a three.js BufferAttribute into a texture. An instance can be treated as a
 * texture and when passing as a uniform to a shader they should be used as a `sampler2d`,
 * `usampler2d`, and `isampler2d` when using the Float, Uint, and Int texture types respectively.
 *
 * _extends THREE.DataTexture_
 *
 * @section Shader and Texture Packing API
 */
class VertexAttributeTexture extends DataTexture {

	constructor() {

		super();
		this.minFilter = NearestFilter;
		this.magFilter = NearestFilter;
		this.generateMipmaps = false;

		/**
		 * Treats `BufferAttribute.itemSize` as though it were set to this value when packing the
		 * buffer attribute texture. Throws an error if the value does not divide evenly into the
		 * length of the BufferAttribute buffer (`count * itemSize % overrideItemSize`).
		 *
		 * Specifically used to pack geometry indices into an RGB texture rather than an Red texture.
		 * @type {number}
		 */
		this.overrideItemSize = null;
		this._forcedType = null;

	}

	/**
	 * Updates the texture to have the data contained in the passed BufferAttribute using the
	 * BufferAttribute `itemSize` field, `normalized` field, and TypedArray layout to determine
	 * the appropriate texture layout, format, and type. The texture dimensions will always be
	 * square. Because these are intended to be sampled as 1D arrays the width of the texture must
	 * be taken into account to derive a sampling uv. See `texelFetch1D` in shaderFunctions.
	 *
	 * @param {BufferAttribute} attribute
	 * @returns {void}
	 */
	updateFrom( attr ) {

		const overrideItemSize = this.overrideItemSize;
		const originalItemSize = attr.itemSize;
		const originalCount = attr.count;
		if ( overrideItemSize !== null ) {

			if ( ( originalItemSize * originalCount ) % overrideItemSize !== 0.0 ) {

				throw new Error( 'VertexAttributeTexture: overrideItemSize must divide evenly into buffer length.' );

			}

			attr.itemSize = overrideItemSize;
			attr.count = originalCount * originalItemSize / overrideItemSize;

		}

		const itemSize = attr.itemSize;
		const count = attr.count;
		const normalized = attr.normalized;
		const originalBufferCons = attr.array.constructor;
		const byteCount = originalBufferCons.BYTES_PER_ELEMENT;
		let targetType = this._forcedType;
		let finalStride = itemSize;

		// derive the type of texture this should be in the shader
		if ( targetType === null ) {

			switch ( originalBufferCons ) {

				case Float32Array:
					targetType = FloatType;
					break;

				case Uint8Array:
				case Uint16Array:
				case Uint32Array:
					targetType = UnsignedIntType;
					break;

				case Int8Array:
				case Int16Array:
				case Int32Array:
					targetType = IntType;
					break;

			}

		}

		// get the target format to store the texture as
		let type, format, normalizeValue, targetBufferCons;
		let internalFormat = countToStringFormat( itemSize );
		switch ( targetType ) {

			case FloatType:
				normalizeValue = 1.0;
				format = countToFormat( itemSize );

				if ( normalized && byteCount === 1 ) {

					targetBufferCons = originalBufferCons;
					internalFormat += '8';

					if ( originalBufferCons === Uint8Array ) {

						type = UnsignedByteType;

					} else {

						type = ByteType;
						internalFormat += '_SNORM';

					}

				} else {

					targetBufferCons = Float32Array;
					internalFormat += '32F';
					type = FloatType;

				}

				break;

			case IntType:
				internalFormat += byteCount * 8 + 'I';
				normalizeValue = normalized ? Math.pow( 2, originalBufferCons.BYTES_PER_ELEMENT * 8 - 1 ) : 1.0;
				format = countToIntFormat( itemSize );

				if ( byteCount === 1 ) {

					targetBufferCons = Int8Array;
					type = ByteType;

				} else if ( byteCount === 2 ) {

					targetBufferCons = Int16Array;
					type = ShortType;

				} else {

					targetBufferCons = Int32Array;
					type = IntType;

				}

				break;

			case UnsignedIntType:
				internalFormat += byteCount * 8 + 'UI';
				normalizeValue = normalized ? Math.pow( 2, originalBufferCons.BYTES_PER_ELEMENT * 8 - 1 ) : 1.0;
				format = countToIntFormat( itemSize );

				if ( byteCount === 1 ) {

					targetBufferCons = Uint8Array;
					type = UnsignedByteType;

				} else if ( byteCount === 2 ) {

					targetBufferCons = Uint16Array;
					type = UnsignedShortType;

				} else {

					targetBufferCons = Uint32Array;
					type = UnsignedIntType;

				}

				break;

		}

		// there will be a mismatch between format length and final length because
		// RGBFormat and RGBIntegerFormat was removed
		if ( finalStride === 3 && ( format === RGBAFormat || format === RGBAIntegerFormat ) ) {

			finalStride = 4;

		}

		// copy the data over to the new texture array
		const dimension = Math.ceil( Math.sqrt( count ) ) || 1;
		const length = finalStride * dimension * dimension;
		const dataArray = new targetBufferCons( length );

		// temporarily set the normalized state to false since we have custom normalization logic
		const originalNormalized = attr.normalized;
		attr.normalized = false;
		for ( let i = 0; i < count; i ++ ) {

			const ii = finalStride * i;
			dataArray[ ii ] = attr.getX( i ) / normalizeValue;

			if ( itemSize >= 2 ) {

				dataArray[ ii + 1 ] = attr.getY( i ) / normalizeValue;

			}

			if ( itemSize >= 3 ) {

				dataArray[ ii + 2 ] = attr.getZ( i ) / normalizeValue;

				if ( finalStride === 4 ) {

					dataArray[ ii + 3 ] = 1.0;

				}

			}

			if ( itemSize >= 4 ) {

				dataArray[ ii + 3 ] = attr.getW( i ) / normalizeValue;

			}

		}

		attr.normalized = originalNormalized;

		this.internalFormat = internalFormat;
		this.format = format;
		this.type = type;
		this.image.width = dimension;
		this.image.height = dimension;
		this.image.data = dataArray;
		this.needsUpdate = true;
		this.dispose();

		attr.itemSize = originalItemSize;
		attr.count = originalCount;

	}

}

/**
 * A VertexAttributeTexture that forces the unsigned integer texture type.
 * @extends VertexAttributeTexture
 * @section Shader and Texture Packing API
 */
class UIntVertexAttributeTexture extends VertexAttributeTexture {

	constructor() {

		super();
		this._forcedType = UnsignedIntType;

	}

}

/**
 * A VertexAttributeTexture that forces the signed integer texture type.
 * @extends VertexAttributeTexture
 * @section Shader and Texture Packing API
 */
class IntVertexAttributeTexture extends VertexAttributeTexture {

	constructor() {

		super();
		this._forcedType = IntType;

	}


}

/**
 * A VertexAttributeTexture that forces the float texture type.
 * @extends VertexAttributeTexture
 * @section Shader and Texture Packing API
 */
class FloatVertexAttributeTexture extends VertexAttributeTexture {

	constructor() {

		super();
		this._forcedType = FloatType;

	}

}

/** @import { MeshBVH } from '../core/MeshBVH.js' */

/**
 * A shader uniform object corresponding to the `BVH` shader struct defined in shaderStructs. The
 * object contains four textures containing information about the BVH and geometry so it can be
 * queried in a shader using the bvh intersection functions defined in shaderFunctions. This object
 * is intended to be used as a shader uniform and read in the shader as a `BVH` struct.
 *
 * @section Shader and Texture Packing API
 */
class MeshBVHUniformStruct {

	constructor() {

		this.index = new UIntVertexAttributeTexture();
		this.position = new FloatVertexAttributeTexture();
		this.bvhBounds = new DataTexture();
		this.bvhContents = new DataTexture();
		this._cachedIndexAttr = null;

		this.index.overrideItemSize = 3;

	}

	/**
	 * Updates the object and associated textures with data from the provided BVH.
	 *
	 * @param {MeshBVH} bvh
	 * @returns {void}
	 */
	updateFrom( bvh ) {

		const { geometry } = bvh;
		bvhToTextures( bvh, this.bvhBounds, this.bvhContents );

		this.position.updateFrom( geometry.attributes.position );

		// dereference a new index attribute if we're using indirect storage
		if ( bvh.indirect ) {

			const indirectBuffer = bvh._indirectBuffer;
			if (
				this._cachedIndexAttr === null ||
				this._cachedIndexAttr.count !== indirectBuffer.length
			) {

				if ( geometry.index ) {

					this._cachedIndexAttr = geometry.index.clone();

				} else {

					const array = getIndexArray( getVertexCount( geometry ) );
					this._cachedIndexAttr = new BufferAttribute( array, 1, false );

				}

			}

			dereferenceIndex( geometry, indirectBuffer, this._cachedIndexAttr );
			this.index.updateFrom( this._cachedIndexAttr );

		} else {

			this.index.updateFrom( geometry.index );

		}

	}

	/**
	 * Dispose of the associated textures.
	 *
	 * @returns {void}
	 */
	dispose() {

		const { index, position, bvhBounds, bvhContents } = this;

		if ( index ) index.dispose();
		if ( position ) position.dispose();
		if ( bvhBounds ) bvhBounds.dispose();
		if ( bvhContents ) bvhContents.dispose();

	}

}

function dereferenceIndex( geometry, indirectBuffer, target ) {

	const unpacked = target.array;
	const indexArray = geometry.index ? geometry.index.array : null;
	for ( let i = 0, l = indirectBuffer.length; i < l; i ++ ) {

		const i3 = 3 * i;
		const v3 = 3 * indirectBuffer[ i ];
		for ( let c = 0; c < 3; c ++ ) {

			unpacked[ i3 + c ] = indexArray ? indexArray[ v3 + c ] : v3 + c;

		}

	}

}

function bvhToTextures( bvh, boundsTexture, contentsTexture ) {

	const roots = bvh._roots;

	if ( roots.length !== 1 ) {

		throw new Error( 'MeshBVHUniformStruct: Multi-root BVHs not supported.' );

	}

	const root = roots[ 0 ];
	const uint16Array = new Uint16Array( root );
	const uint32Array = new Uint32Array( root );
	const float32Array = new Float32Array( root );

	// Both bounds need two elements per node so compute the height so it's twice as long as
	// the width so we can expand the row by two and still have a square texture
	const nodeCount = root.byteLength / BYTES_PER_NODE;
	const boundsDimension = 2 * Math.ceil( Math.sqrt( nodeCount / 2 ) );
	const boundsArray = new Float32Array( 4 * boundsDimension * boundsDimension );

	const contentsDimension = Math.ceil( Math.sqrt( nodeCount ) );
	const contentsArray = new Uint32Array( 2 * contentsDimension * contentsDimension );

	for ( let i = 0; i < nodeCount; i ++ ) {

		const nodeIndex32 = i * BYTES_PER_NODE / 4;
		const nodeIndex16 = nodeIndex32 * 2;
		const boundsIndex = BOUNDING_DATA_INDEX( nodeIndex32 );
		for ( let b = 0; b < 3; b ++ ) {

			boundsArray[ 8 * i + 0 + b ] = float32Array[ boundsIndex + 0 + b ];
			boundsArray[ 8 * i + 4 + b ] = float32Array[ boundsIndex + 3 + b ];

		}

		if ( IS_LEAF( nodeIndex16, uint16Array ) ) {

			const count = COUNT( nodeIndex16, uint16Array );
			const offset = OFFSET( nodeIndex32, uint32Array );

			const mergedLeafCount = LEAFNODE_MASK_32 | count;
			contentsArray[ i * 2 + 0 ] = mergedLeafCount;
			contentsArray[ i * 2 + 1 ] = offset;

		} else {

			const rightNodeIndex = uint32Array[ nodeIndex32 + 6 ];
			const splitAxis = SPLIT_AXIS( nodeIndex32, uint32Array );

			contentsArray[ i * 2 + 0 ] = splitAxis;
			contentsArray[ i * 2 + 1 ] = rightNodeIndex;

		}

	}

	boundsTexture.image.data = boundsArray;
	boundsTexture.image.width = boundsDimension;
	boundsTexture.image.height = boundsDimension;
	boundsTexture.format = RGBAFormat;
	boundsTexture.type = FloatType;
	boundsTexture.internalFormat = 'RGBA32F';
	boundsTexture.minFilter = NearestFilter;
	boundsTexture.magFilter = NearestFilter;
	boundsTexture.generateMipmaps = false;
	boundsTexture.needsUpdate = true;
	boundsTexture.dispose();

	contentsTexture.image.data = contentsArray;
	contentsTexture.image.width = contentsDimension;
	contentsTexture.image.height = contentsDimension;
	contentsTexture.format = RGIntegerFormat;
	contentsTexture.type = UnsignedIntType;
	contentsTexture.internalFormat = 'RG32UI';
	contentsTexture.minFilter = NearestFilter;
	contentsTexture.magFilter = NearestFilter;
	contentsTexture.generateMipmaps = false;
	contentsTexture.needsUpdate = true;
	contentsTexture.dispose();

}

/** @import { Mesh, Material, Object3D } from 'three' */

const _positionVector = /*@__PURE__*/ new Vector3();
const _normalVector = /*@__PURE__*/ new Vector3();
const _tangentVector = /*@__PURE__*/ new Vector3();
const _tangentVector4 = /*@__PURE__*/ new Vector4();

const _morphVector = /*@__PURE__*/ new Vector3();
const _temp = /*@__PURE__*/ new Vector3();

const _skinIndex = /*@__PURE__*/ new Vector4();
const _skinWeight = /*@__PURE__*/ new Vector4();
const _matrix = /*@__PURE__*/ new Matrix4();
const _boneMatrix = /*@__PURE__*/ new Matrix4();

// Confirms that the two provided attributes are compatible
function validateAttributes( attr1, attr2 ) {

	if ( ! attr1 && ! attr2 ) {

		return;

	}

	const sameCount = attr1.count === attr2.count;
	const sameNormalized = attr1.normalized === attr2.normalized;
	const sameType = attr1.array.constructor === attr2.array.constructor;
	const sameItemSize = attr1.itemSize === attr2.itemSize;

	if ( ! sameCount || ! sameNormalized || ! sameType || ! sameItemSize ) {

		throw new Error();

	}

}

// Clones the given attribute with a new compatible buffer attribute but no data
function createAttributeClone( attr, countOverride = null ) {

	const cons = attr.array.constructor;
	const normalized = attr.normalized;
	const itemSize = attr.itemSize;
	const count = countOverride === null ? attr.count : countOverride;

	return new BufferAttribute( new cons( itemSize * count ), itemSize, normalized );

}

// target offset is the number of elements in the target buffer stride to skip before copying the
// attributes contents in to.
function copyAttributeContents( attr, target, targetOffset = 0 ) {

	if ( attr.isInterleavedBufferAttribute ) {

		const itemSize = attr.itemSize;
		for ( let i = 0, l = attr.count; i < l; i ++ ) {

			const io = i + targetOffset;
			target.setX( io, attr.getX( i ) );
			if ( itemSize >= 2 ) target.setY( io, attr.getY( i ) );
			if ( itemSize >= 3 ) target.setZ( io, attr.getZ( i ) );
			if ( itemSize >= 4 ) target.setW( io, attr.getW( i ) );

		}

	} else {

		const array = target.array;
		const cons = array.constructor;
		const byteOffset = array.BYTES_PER_ELEMENT * attr.itemSize * targetOffset;
		const temp = new cons( array.buffer, byteOffset, attr.array.length );
		temp.set( attr.array );

	}

}

// Adds the "matrix" multiplied by "scale" to "target"
function addScaledMatrix( target, matrix, scale ) {

	const targetArray = target.elements;
	const matrixArray = matrix.elements;
	for ( let i = 0, l = matrixArray.length; i < l; i ++ ) {

		targetArray[ i ] += matrixArray[ i ] * scale;

	}

}

// A version of "SkinnedMesh.boneTransform" for normals
function boneNormalTransform( mesh, index, target ) {

	const skeleton = mesh.skeleton;
	const geometry = mesh.geometry;
	const bones = skeleton.bones;
	const boneInverses = skeleton.boneInverses;

	_skinIndex.fromBufferAttribute( geometry.attributes.skinIndex, index );
	_skinWeight.fromBufferAttribute( geometry.attributes.skinWeight, index );

	_matrix.elements.fill( 0 );

	for ( let i = 0; i < 4; i ++ ) {

		const weight = _skinWeight.getComponent( i );

		if ( weight !== 0 ) {

			const boneIndex = _skinIndex.getComponent( i );
			_boneMatrix.multiplyMatrices( bones[ boneIndex ].matrixWorld, boneInverses[ boneIndex ] );

			addScaledMatrix( _matrix, _boneMatrix, weight );

		}

	}

	_matrix.multiply( mesh.bindMatrix ).premultiply( mesh.bindMatrixInverse );
	target.transformDirection( _matrix );

	return target;

}

// Applies the morph target data to the target vector
function applyMorphTarget( morphData, morphInfluences, morphTargetsRelative, i, target ) {

	_morphVector.set( 0, 0, 0 );
	for ( let j = 0, jl = morphData.length; j < jl; j ++ ) {

		const influence = morphInfluences[ j ];
		const morphAttribute = morphData[ j ];

		if ( influence === 0 ) continue;

		_temp.fromBufferAttribute( morphAttribute, i );

		if ( morphTargetsRelative ) {

			_morphVector.addScaledVector( _temp, influence );

		} else {

			_morphVector.addScaledVector( _temp.sub( target ), influence );

		}

	}

	target.add( _morphVector );

}

// Modified version of BufferGeometryUtils.mergeBufferGeometries that ignores morph targets and updates a attributes in place
function mergeBufferGeometries( geometries, options = { useGroups: false, updateIndex: false, skipAttributes: [] }, targetGeometry = new BufferGeometry() ) {

	const isIndexed = geometries[ 0 ].index !== null;
	const { useGroups = false, updateIndex = false, skipAttributes = [] } = options;

	const attributesUsed = new Set( Object.keys( geometries[ 0 ].attributes ) );
	const attributes = {};

	let offset = 0;

	targetGeometry.clearGroups();
	for ( let i = 0; i < geometries.length; ++ i ) {

		const geometry = geometries[ i ];
		let attributesCount = 0;

		// ensure that all geometries are indexed, or none
		if ( isIndexed !== ( geometry.index !== null ) ) {

			throw new Error( 'StaticGeometryGenerator: All geometries must have compatible attributes; make sure index attribute exists among all geometries, or in none of them.' );

		}

		// gather attributes, exit early if they're different
		for ( const name in geometry.attributes ) {

			if ( ! attributesUsed.has( name ) ) {

				throw new Error( 'StaticGeometryGenerator: All geometries must have compatible attributes; make sure "' + name + '" attribute exists among all geometries, or in none of them.' );

			}

			if ( attributes[ name ] === undefined ) {

				attributes[ name ] = [];

			}

			attributes[ name ].push( geometry.attributes[ name ] );
			attributesCount ++;

		}

		// ensure geometries have the same number of attributes
		if ( attributesCount !== attributesUsed.size ) {

			throw new Error( 'StaticGeometryGenerator: Make sure all geometries have the same number of attributes.' );

		}

		if ( useGroups ) {

			let count;
			if ( isIndexed ) {

				count = geometry.index.count;

			} else if ( geometry.attributes.position !== undefined ) {

				count = geometry.attributes.position.count;

			} else {

				throw new Error( 'StaticGeometryGenerator: The geometry must have either an index or a position attribute' );

			}

			targetGeometry.addGroup( offset, count, i );
			offset += count;

		}

	}

	// merge indices
	if ( isIndexed ) {

		let forceUpdateIndex = false;
		if ( ! targetGeometry.index ) {

			let indexCount = 0;
			for ( let i = 0; i < geometries.length; ++ i ) {

				indexCount += geometries[ i ].index.count;

			}

			targetGeometry.setIndex( new BufferAttribute( new Uint32Array( indexCount ), 1, false ) );
			forceUpdateIndex = true;

		}

		if ( updateIndex || forceUpdateIndex ) {

			const targetIndex = targetGeometry.index;
			let targetOffset = 0;
			let indexOffset = 0;
			for ( let i = 0; i < geometries.length; ++ i ) {

				const geometry = geometries[ i ];
				const index = geometry.index;
				if ( skipAttributes[ i ] !== true ) {

					for ( let j = 0; j < index.count; ++ j ) {

						targetIndex.setX( targetOffset, index.getX( j ) + indexOffset );
						targetOffset ++;

					}

				}

				indexOffset += geometry.attributes.position.count;

			}

		}

	}

	// merge attributes
	for ( const name in attributes ) {

		const attrList = attributes[ name ];
		if ( ! ( name in targetGeometry.attributes ) ) {

			let count = 0;
			for ( const key in attrList ) {

				count += attrList[ key ].count;

			}

			targetGeometry.setAttribute( name, createAttributeClone( attributes[ name ][ 0 ], count ) );

		}

		const targetAttribute = targetGeometry.attributes[ name ];
		let offset = 0;
		for ( let i = 0, l = attrList.length; i < l; i ++ ) {

			const attr = attrList[ i ];
			if ( skipAttributes[ i ] !== true ) {

				copyAttributeContents( attr, targetAttribute, offset );

			}

			offset += attr.count;

		}

	}

	return targetGeometry;

}

function checkTypedArrayEquality( a, b ) {

	if ( a === null || b === null ) {

		return a === b;

	}

	if ( a.length !== b.length ) {

		return false;

	}

	for ( let i = 0, l = a.length; i < l; i ++ ) {

		if ( a[ i ] !== b[ i ] ) {

			return false;

		}

	}

	return true;

}

function invertGeometry( geometry ) {

	const { index, attributes } = geometry;
	if ( index ) {

		for ( let i = 0, l = index.count; i < l; i += 3 ) {

			const v0 = index.getX( i );
			const v2 = index.getX( i + 2 );
			index.setX( i, v2 );
			index.setX( i + 2, v0 );

		}

	} else {

		for ( const key in attributes ) {

			const attr = attributes[ key ];
			const itemSize = attr.itemSize;
			for ( let i = 0, l = attr.count; i < l; i += 3 ) {

				for ( let j = 0; j < itemSize; j ++ ) {

					const v0 = attr.getComponent( i, j );
					const v2 = attr.getComponent( i + 2, j );
					attr.setComponent( i, j, v2 );
					attr.setComponent( i + 2, j, v0 );

				}

			}

		}

	}

	return geometry;


}

// Checks whether the geometry changed between this and last evaluation
class GeometryDiff {

	constructor( mesh ) {

		this.matrixWorld = new Matrix4();
		this.geometryHash = null;
		this.boneMatrices = null;
		this.primitiveCount = - 1;
		this.mesh = mesh;

		this.update();

	}

	update() {

		const mesh = this.mesh;
		const geometry = mesh.geometry;
		const skeleton = mesh.skeleton;
		const primitiveCount = ( geometry.index ? geometry.index.count : geometry.attributes.position.count ) / 3;
		this.matrixWorld.copy( mesh.matrixWorld );
		this.geometryHash = geometry.attributes.position.version;
		this.primitiveCount = primitiveCount;

		if ( skeleton ) {

			// ensure the bone matrix array is updated to the appropriate length
			if ( ! skeleton.boneTexture ) {

				skeleton.computeBoneTexture();

			}

			skeleton.update();

			// copy data if possible otherwise clone it
			const boneMatrices = skeleton.boneMatrices;
			if ( ! this.boneMatrices || this.boneMatrices.length !== boneMatrices.length ) {

				this.boneMatrices = boneMatrices.slice();

			} else {

				this.boneMatrices.set( boneMatrices );

			}

		} else {

			this.boneMatrices = null;

		}

	}

	didChange() {

		const mesh = this.mesh;
		const geometry = mesh.geometry;
		const primitiveCount = ( geometry.index ? geometry.index.count : geometry.attributes.position.count ) / 3;
		const identical =
			this.matrixWorld.equals( mesh.matrixWorld ) &&
			this.geometryHash === geometry.attributes.position.version &&
			checkTypedArrayEquality( mesh.skeleton && mesh.skeleton.boneMatrices || null, this.boneMatrices ) &&
			this.primitiveCount === primitiveCount;

		return ! identical;

	}

}

/**
 * A utility class for taking a set of SkinnedMeshes or morph target geometry and baking it into
 * a single, static geometry that a BVH can be generated for.
 */
class StaticGeometryGenerator {

	/**
	 * Takes an array of object hierarchies to bake into a single static geometry.
	 *
	 * @param {Object3D|Array<Object3D>} meshes
	 */
	constructor( meshes ) {

		if ( ! Array.isArray( meshes ) ) {

			meshes = [ meshes ];

		}

		const finalMeshes = [];
		meshes.forEach( object => {

			object.traverseVisible( c => {

				if ( c.isMesh ) {

					finalMeshes.push( c );

				}

			} );

		} );

		/**
		 * @type {Array<Mesh>}
		 */
		this.meshes = finalMeshes;

		/**
		 * If true then groups are used to support an array of materials on the mesh.
		 * @type {boolean}
		 */
		this.useGroups = true;

		/**
		 * Whether to transform the vertices of the geometry by the world transforms of each mesh when generating.
		 * @type {boolean}
		 */
		this.applyWorldTransforms = true;

		/**
		 * The set of attributes to copy onto the static geometry.
		 * @type {Array<string>}
		 */
		this.attributes = [ 'position', 'normal', 'color', 'tangent', 'uv', 'uv2' ];
		this._intermediateGeometry = new Array( finalMeshes.length ).fill().map( () => new BufferGeometry() );
		this._diffMap = new WeakMap();

	}

	/**
	 * Returns an array of materials for the meshes to be merged. These can be used alongside the
	 * generated geometry when creating a mesh: `new Mesh( geometry, generator.getMaterials() )`.
	 *
	 * @returns {Array<Material>}
	 */
	getMaterials() {

		const materials = [];
		this.meshes.forEach( mesh => {

			if ( Array.isArray( mesh.material ) ) {

				materials.push( ...mesh.material );

			} else {

				materials.push( mesh.material );

			}

		} );
		return materials;

	}

	/**
	 * Generates a single, static geometry for the passed meshes. When generating for the first
	 * time an empty target geometry is expected. The same generated geometry can be passed into
	 * the function on subsequent calls to update the geometry in place to save memory. An error
	 * will be thrown if the attributes or geometry on the meshes to bake has been changed and
	 * are incompatible lengths, types, etc.
	 *
	 * On subsequent calls the "index" buffer will not be modified so any BVH generated for the
	 * geometry is unaffected. Once the geometry is updated the `MeshBVH.refit` function can be
	 * called to update the BVH.
	 *
	 * @param {BufferGeometry} [targetGeometry]
	 * @returns {BufferGeometry}
	 */
	generate( targetGeometry = new BufferGeometry() ) {

		// track which attributes have been updated and which to skip to avoid unnecessary attribute copies
		let skipAttributes = [];
		const { meshes, useGroups, _intermediateGeometry, _diffMap } = this;
		for ( let i = 0, l = meshes.length; i < l; i ++ ) {

			const mesh = meshes[ i ];
			const geom = _intermediateGeometry[ i ];
			const diff = _diffMap.get( mesh );
			if ( ! diff || diff.didChange( mesh ) ) {

				this._convertToStaticGeometry( mesh, geom );
				skipAttributes.push( false );

				if ( ! diff ) {

					_diffMap.set( mesh, new GeometryDiff( mesh ) );

				} else {

					diff.update();

				}

			} else {

				skipAttributes.push( true );

			}

		}

		if ( _intermediateGeometry.length === 0 ) {

			// if there are no geometries then just create a fake empty geometry to provide
			targetGeometry.setIndex( null );

			// remove all geometry
			const attrs = targetGeometry.attributes;
			for ( const key in attrs ) {

				targetGeometry.deleteAttribute( key );

			}

			// create dummy attributes
			for ( const key in this.attributes ) {

				targetGeometry.setAttribute( this.attributes[ key ], new BufferAttribute( new Float32Array( 0 ), 4, false ) );

			}

		} else {

			mergeBufferGeometries( _intermediateGeometry, { useGroups, skipAttributes }, targetGeometry );

		}

		for ( const key in targetGeometry.attributes ) {

			targetGeometry.attributes[ key ].needsUpdate = true;

		}

		return targetGeometry;

	}

	_convertToStaticGeometry( mesh, targetGeometry = new BufferGeometry() ) {

		const geometry = mesh.geometry;
		const applyWorldTransforms = this.applyWorldTransforms;
		const includeNormal = this.attributes.includes( 'normal' );
		const includeTangent = this.attributes.includes( 'tangent' );
		const attributes = geometry.attributes;
		const targetAttributes = targetGeometry.attributes;

		// initialize the attributes if they don't exist
		if ( ! targetGeometry.index && geometry.index ) {

			targetGeometry.index = geometry.index.clone();

		}

		if ( ! targetAttributes.position ) {

			targetGeometry.setAttribute( 'position', createAttributeClone( attributes.position ) );

		}

		if ( includeNormal && ! targetAttributes.normal && attributes.normal ) {

			targetGeometry.setAttribute( 'normal', createAttributeClone( attributes.normal ) );

		}

		if ( includeTangent && ! targetAttributes.tangent && attributes.tangent ) {

			targetGeometry.setAttribute( 'tangent', createAttributeClone( attributes.tangent ) );

		}

		// ensure the attributes are consistent
		validateAttributes( geometry.index, targetGeometry.index );
		validateAttributes( attributes.position, targetAttributes.position );

		if ( includeNormal ) {

			validateAttributes( attributes.normal, targetAttributes.normal );

		}

		if ( includeTangent ) {

			validateAttributes( attributes.tangent, targetAttributes.tangent );

		}

		// generate transformed vertex attribute data
		const position = attributes.position;
		const normal = includeNormal ? attributes.normal : null;
		const tangent = includeTangent ? attributes.tangent : null;
		const morphPosition = geometry.morphAttributes.position;
		const morphNormal = geometry.morphAttributes.normal;
		const morphTangent = geometry.morphAttributes.tangent;
		const morphTargetsRelative = geometry.morphTargetsRelative;
		const morphInfluences = mesh.morphTargetInfluences;
		const normalMatrix = new Matrix3();
		normalMatrix.getNormalMatrix( mesh.matrixWorld );

		// copy the index
		if ( geometry.index ) {

			targetGeometry.index.array.set( geometry.index.array );

		}

		// copy and apply other attributes
		for ( let i = 0, l = attributes.position.count; i < l; i ++ ) {

			_positionVector.fromBufferAttribute( position, i );
			if ( normal ) {

				_normalVector.fromBufferAttribute( normal, i );

			}

			if ( tangent ) {

				_tangentVector4.fromBufferAttribute( tangent, i );
				_tangentVector.fromBufferAttribute( tangent, i );

			}

			// apply morph target transform
			if ( morphInfluences ) {

				if ( morphPosition ) {

					applyMorphTarget( morphPosition, morphInfluences, morphTargetsRelative, i, _positionVector );

				}

				if ( morphNormal ) {

					applyMorphTarget( morphNormal, morphInfluences, morphTargetsRelative, i, _normalVector );

				}

				if ( morphTangent ) {

					applyMorphTarget( morphTangent, morphInfluences, morphTargetsRelative, i, _tangentVector );

				}

			}

			// apply bone transform
			if ( mesh.isSkinnedMesh ) {

				mesh.applyBoneTransform( i, _positionVector );
				if ( normal ) {

					boneNormalTransform( mesh, i, _normalVector );

				}

				if ( tangent ) {

					boneNormalTransform( mesh, i, _tangentVector );

				}

			}

			// update the vectors of the attributes
			if ( applyWorldTransforms ) {

				_positionVector.applyMatrix4( mesh.matrixWorld );

			}

			targetAttributes.position.setXYZ( i, _positionVector.x, _positionVector.y, _positionVector.z );

			if ( normal ) {

				if ( applyWorldTransforms ) {

					_normalVector.applyNormalMatrix( normalMatrix );

				}

				targetAttributes.normal.setXYZ( i, _normalVector.x, _normalVector.y, _normalVector.z );

			}

			if ( tangent ) {

				if ( applyWorldTransforms ) {

					_tangentVector.transformDirection( mesh.matrixWorld );

				}

				targetAttributes.tangent.setXYZW( i, _tangentVector.x, _tangentVector.y, _tangentVector.z, _tangentVector4.w );

			}

		}

		// copy other attributes over
		for ( const i in this.attributes ) {

			const key = this.attributes[ i ];
			if ( key === 'position' || key === 'tangent' || key === 'normal' || ! ( key in attributes ) ) {

				continue;

			}

			if ( ! targetAttributes[ key ] ) {

				targetGeometry.setAttribute( key, createAttributeClone( attributes[ key ] ) );

			}

			validateAttributes( attributes[ key ], targetAttributes[ key ] );
			copyAttributeContents( attributes[ key ], targetAttributes[ key ] );

		}

		if ( mesh.matrixWorld.determinant() < 0 ) {

			invertGeometry( targetGeometry );

		}

		return targetGeometry;

	}

}

/**
 * Set of shader functions used for interacting with the packed BVH in a shader and sampling
 * VertexAttributeTextures. Provides common utility functions including `texelFetch1D`. See
 * [src/webgl/glsl](https://github.com/gkjohnson/three-mesh-bvh/tree/master/src/webgl/glsl)
 * for full implementations and declarations.
 *
 * Accessed as `BVHShaderGLSL.common_functions`.
 *
 * @section Shader and Texture Packing API
 * @type {string}
 */
const common_functions = /* glsl */`

// A stack of uint32 indices can can store the indices for
// a perfectly balanced tree with a depth up to 31. Lower stack
// depth gets higher performance.
//
// However not all trees are balanced. Best value to set this to
// is the trees max depth.
#ifndef BVH_STACK_DEPTH
#define BVH_STACK_DEPTH 60
#endif

#ifndef INFINITY
#define INFINITY 1e20
#endif

// Utilities
uvec4 uTexelFetch1D( usampler2D tex, uint index ) {

	uint width = uint( textureSize( tex, 0 ).x );
	uvec2 uv;
	uv.x = index % width;
	uv.y = index / width;

	return texelFetch( tex, ivec2( uv ), 0 );

}

ivec4 iTexelFetch1D( isampler2D tex, uint index ) {

	uint width = uint( textureSize( tex, 0 ).x );
	uvec2 uv;
	uv.x = index % width;
	uv.y = index / width;

	return texelFetch( tex, ivec2( uv ), 0 );

}

vec4 texelFetch1D( sampler2D tex, uint index ) {

	uint width = uint( textureSize( tex, 0 ).x );
	uvec2 uv;
	uv.x = index % width;
	uv.y = index / width;

	return texelFetch( tex, ivec2( uv ), 0 );

}

vec4 textureSampleBarycoord( sampler2D tex, vec3 barycoord, uvec3 faceIndices ) {

	return
		barycoord.x * texelFetch1D( tex, faceIndices.x ) +
		barycoord.y * texelFetch1D( tex, faceIndices.y ) +
		barycoord.z * texelFetch1D( tex, faceIndices.z );

}

void ndcToCameraRay(
	vec2 coord, mat4 cameraWorld, mat4 invProjectionMatrix,
	out vec3 rayOrigin, out vec3 rayDirection
) {

	// get camera look direction and near plane for camera clipping
	vec4 lookDirection = cameraWorld * vec4( 0.0, 0.0, - 1.0, 0.0 );
	vec4 nearVector = invProjectionMatrix * vec4( 0.0, 0.0, - 1.0, 1.0 );
	float near = abs( nearVector.z / nearVector.w );

	// get the camera direction and position from camera matrices
	vec4 origin = cameraWorld * vec4( 0.0, 0.0, 0.0, 1.0 );
	vec4 direction = invProjectionMatrix * vec4( coord, 0.5, 1.0 );
	direction /= direction.w;
	direction = cameraWorld * direction - origin;

	// slide the origin along the ray until it sits at the near clip plane position
	origin.xyz += direction.xyz * near / dot( direction, lookDirection );

	rayOrigin = origin.xyz;
	rayDirection = direction.xyz;

}
`;

// Distance to Point

/**
 * Set of shader functions used for interacting with the packed BVH in a shader and sampling
 * VertexAttributeTextures. Provides distance query functions. See
 * [src/webgl/glsl](https://github.com/gkjohnson/three-mesh-bvh/tree/master/src/webgl/glsl)
 * for full implementations and declarations.
 *
 * Accessed as `BVHShaderGLSL.bvh_distance_functions`.
 *
 * @section Shader and Texture Packing API
 * @type {string}
 */
const bvh_distance_functions = /* glsl */`

float dot2( vec3 v ) {

	return dot( v, v );

}

// implementation from https://www.shadertoy.com/view/ttfGWl, though method 2 has been removed
// and is now available at this fork: https://www.shadertoy.com/view/WlB3zW
vec3 closestPointToTriangle( vec3 p, vec3 v0, vec3 v1, vec3 v2, out vec3 barycoord ) {

    vec3 v10 = v1 - v0;
    vec3 v21 = v2 - v1;
    vec3 v02 = v0 - v2;

	vec3 p0 = p - v0;
	vec3 p1 = p - v1;
	vec3 p2 = p - v2;

    vec3 nor = cross( v10, v02 );

    // method 2, in barycentric space
    vec3  q = cross( nor, p0 );
    float d = 1.0 / dot2( nor );
    float u = d * dot( q, v02 );
    float v = d * dot( q, v10 );
    float w = 1.0 - u - v;

	if( u < 0.0 ) {

		w = clamp( dot( p2, v02 ) / dot2( v02 ), 0.0, 1.0 );
		u = 0.0;
		v = 1.0 - w;

	} else if( v < 0.0 ) {

		u = clamp( dot( p0, v10 ) / dot2( v10 ), 0.0, 1.0 );
		v = 0.0;
		w = 1.0 - u;

	} else if( w < 0.0 ) {

		v = clamp( dot( p1, v21 ) / dot2( v21 ), 0.0, 1.0 );
		w = 0.0;
		u = 1.0 - v;

	}

	// output the barycoord in v0, v1, v2 weight order
	barycoord = vec3( w, u, v );
    return u * v1 + v * v2 + w * v0;

}

float distanceToTriangles(
	// geometry info and triangle range
	sampler2D positionAttr, usampler2D indexAttr, uint offset, uint count,

	// point and cut off range
	vec3 point, float closestDistanceSquared,

	// outputs
	inout uvec4 faceIndices, inout vec3 faceNormal, inout vec3 barycoord, inout float side, inout vec3 outPoint
) {

	bool found = false;
	vec3 localBarycoord;
	for ( uint i = offset, l = offset + count; i < l; i ++ ) {

		uvec3 indices = uTexelFetch1D( indexAttr, i ).xyz;
		vec3 a = texelFetch1D( positionAttr, indices.x ).rgb;
		vec3 b = texelFetch1D( positionAttr, indices.y ).rgb;
		vec3 c = texelFetch1D( positionAttr, indices.z ).rgb;

		// get the closest point and barycoord
		vec3 closestPoint = closestPointToTriangle( point, a, b, c, localBarycoord );
		vec3 delta = point - closestPoint;
		float sqDist = dot2( delta );
		if ( sqDist < closestDistanceSquared ) {

			// set the output results
			closestDistanceSquared = sqDist;
			faceIndices = uvec4( indices.xyz, i );
			faceNormal = normalize( cross( a - b, b - c ) );
			barycoord = localBarycoord;
			outPoint = closestPoint;
			side = sign( dot( faceNormal, delta ) );

		}

	}

	return closestDistanceSquared;

}

float distanceSqToBounds( vec3 point, vec3 boundsMin, vec3 boundsMax ) {

	vec3 clampedPoint = clamp( point, boundsMin, boundsMax );
	vec3 delta = point - clampedPoint;
	return dot( delta, delta );

}

float distanceSqToBVHNodeBoundsPoint( vec3 point, sampler2D bvhBounds, uint currNodeIndex ) {

	uint cni2 = currNodeIndex * 2u;
	vec3 boundsMin = texelFetch1D( bvhBounds, cni2 ).xyz;
	vec3 boundsMax = texelFetch1D( bvhBounds, cni2 + 1u ).xyz;
	return distanceSqToBounds( point, boundsMin, boundsMax );

}

// use a macro to hide the fact that we need to expand the struct into separate fields
#define\
	bvhClosestPointToPoint(\
		bvh,\
		point, maxDistance, faceIndices, faceNormal, barycoord, side, outPoint\
	)\
	_bvhClosestPointToPoint(\
		bvh.position, bvh.index, bvh.bvhBounds, bvh.bvhContents,\
		point, maxDistance, faceIndices, faceNormal, barycoord, side, outPoint\
	)

float _bvhClosestPointToPoint(
	// bvh info
	sampler2D bvh_position, usampler2D bvh_index, sampler2D bvh_bvhBounds, usampler2D bvh_bvhContents,

	// point to check
	vec3 point, float maxDistance,

	// output variables
	inout uvec4 faceIndices, inout vec3 faceNormal, inout vec3 barycoord,
	inout float side, inout vec3 outPoint
 ) {

	// stack needs to be twice as long as the deepest tree we expect because
	// we push both the left and right child onto the stack every traversal
	int pointer = 0;
	uint stack[ BVH_STACK_DEPTH ];
	stack[ 0 ] = 0u;

	float closestDistanceSquared = maxDistance * maxDistance;
	bool found = false;
	while ( pointer > - 1 && pointer < BVH_STACK_DEPTH ) {

		uint currNodeIndex = stack[ pointer ];
		pointer --;

		// check if we intersect the current bounds
		float boundsHitDistance = distanceSqToBVHNodeBoundsPoint( point, bvh_bvhBounds, currNodeIndex );
		if ( boundsHitDistance > closestDistanceSquared ) {

			continue;

		}

		uvec2 boundsInfo = uTexelFetch1D( bvh_bvhContents, currNodeIndex ).xy;
		bool isLeaf = bool( boundsInfo.x & 0xffff0000u );
		if ( isLeaf ) {

			uint count = boundsInfo.x & 0x0000ffffu;
			uint offset = boundsInfo.y;
			closestDistanceSquared = distanceToTriangles(
				bvh_position, bvh_index, offset, count, point, closestDistanceSquared,

				// outputs
				faceIndices, faceNormal, barycoord, side, outPoint
			);

		} else {

			uint leftIndex = currNodeIndex + 1u;
			uint splitAxis = boundsInfo.x & 0x0000ffffu;
			uint rightIndex = currNodeIndex + boundsInfo.y;
			bool leftToRight = distanceSqToBVHNodeBoundsPoint( point, bvh_bvhBounds, leftIndex ) < distanceSqToBVHNodeBoundsPoint( point, bvh_bvhBounds, rightIndex );//rayDirection[ splitAxis ] >= 0.0;
			uint c1 = leftToRight ? leftIndex : rightIndex;
			uint c2 = leftToRight ? rightIndex : leftIndex;

			// set c2 in the stack so we traverse it later. We need to keep track of a pointer in
			// the stack while we traverse. The second pointer added is the one that will be
			// traversed first
			pointer ++;
			stack[ pointer ] = c2;
			pointer ++;
			stack[ pointer ] = c1;

		}

	}

	return sqrt( closestDistanceSquared );

}
`;

/**
 * Set of shader functions used for interacting with the packed BVH in a shader and sampling
 * VertexAttributeTextures. Provides ray intersection functions. See
 * [src/webgl/glsl](https://github.com/gkjohnson/three-mesh-bvh/tree/master/src/webgl/glsl)
 * for full implementations and declarations.
 *
 * Accessed as `BVHShaderGLSL.bvh_ray_functions`.
 *
 * @section Shader and Texture Packing API
 * @type {string}
 */
const bvh_ray_functions = /* glsl */`

#ifndef TRI_INTERSECT_EPSILON
#define TRI_INTERSECT_EPSILON 1e-5
#endif

// Raycasting
bool intersectsBounds( vec3 rayOrigin, vec3 rayDirection, vec3 boundsMin, vec3 boundsMax, out float dist ) {

	// https://www.reddit.com/r/opengl/comments/8ntzz5/fast_glsl_ray_box_intersection/
	// https://tavianator.com/2011/ray_box.html
	vec3 invDir = 1.0 / rayDirection;

	// find intersection distances for each plane
	vec3 tMinPlane = invDir * ( boundsMin - rayOrigin );
	vec3 tMaxPlane = invDir * ( boundsMax - rayOrigin );

	// get the min and max distances from each intersection
	vec3 tMinHit = min( tMaxPlane, tMinPlane );
	vec3 tMaxHit = max( tMaxPlane, tMinPlane );

	// get the furthest hit distance
	vec2 t = max( tMinHit.xx, tMinHit.yz );
	float t0 = max( t.x, t.y );

	// get the minimum hit distance
	t = min( tMaxHit.xx, tMaxHit.yz );
	float t1 = min( t.x, t.y );

	// set distance to 0.0 if the ray starts inside the box
	dist = max( t0, 0.0 );

	return t1 >= dist;

}

bool intersectsTriangle(
	vec3 rayOrigin, vec3 rayDirection, vec3 a, vec3 b, vec3 c,
	out vec3 barycoord, out vec3 norm, out float dist, out float side
) {

	// https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
	vec3 edge1 = b - a;
	vec3 edge2 = c - a;
	norm = cross( edge1, edge2 );

	float det = - dot( rayDirection, norm );
	float invdet = 1.0 / det;

	vec3 AO = rayOrigin - a;
	vec3 DAO = cross( AO, rayDirection );

	vec4 uvt;
	uvt.x = dot( edge2, DAO ) * invdet;
	uvt.y = - dot( edge1, DAO ) * invdet;
	uvt.z = dot( AO, norm ) * invdet;
	uvt.w = 1.0 - uvt.x - uvt.y;

	// set the hit information
	barycoord = uvt.wxy; // arranged in A, B, C order
	dist = uvt.z;
	side = sign( det );
	norm = side * normalize( norm );

	// add an epsilon to avoid misses between triangles
	uvt += vec4( TRI_INTERSECT_EPSILON );

	return all( greaterThanEqual( uvt, vec4( 0.0 ) ) );

}

bool intersectTriangles(
	// geometry info and triangle range
	sampler2D positionAttr, usampler2D indexAttr, uint offset, uint count,

	// ray
	vec3 rayOrigin, vec3 rayDirection,

	// outputs
	inout float minDistance, inout uvec4 faceIndices, inout vec3 faceNormal, inout vec3 barycoord,
	inout float side, inout float dist
) {

	bool found = false;
	vec3 localBarycoord, localNormal;
	float localDist, localSide;
	for ( uint i = offset, l = offset + count; i < l; i ++ ) {

		uvec3 indices = uTexelFetch1D( indexAttr, i ).xyz;
		vec3 a = texelFetch1D( positionAttr, indices.x ).rgb;
		vec3 b = texelFetch1D( positionAttr, indices.y ).rgb;
		vec3 c = texelFetch1D( positionAttr, indices.z ).rgb;

		if (
			intersectsTriangle( rayOrigin, rayDirection, a, b, c, localBarycoord, localNormal, localDist, localSide )
			&& localDist < minDistance
		) {

			found = true;
			minDistance = localDist;

			faceIndices = uvec4( indices.xyz, i );
			faceNormal = localNormal;

			side = localSide;
			barycoord = localBarycoord;
			dist = localDist;

		}

	}

	return found;

}

bool intersectsBVHNodeBounds( vec3 rayOrigin, vec3 rayDirection, sampler2D bvhBounds, uint currNodeIndex, out float dist ) {

	uint cni2 = currNodeIndex * 2u;
	vec3 boundsMin = texelFetch1D( bvhBounds, cni2 ).xyz;
	vec3 boundsMax = texelFetch1D( bvhBounds, cni2 + 1u ).xyz;
	return intersectsBounds( rayOrigin, rayDirection, boundsMin, boundsMax, dist );

}

// use a macro to hide the fact that we need to expand the struct into separate fields
#define\
	bvhIntersectFirstHit(\
		bvh,\
		rayOrigin, rayDirection, faceIndices, faceNormal, barycoord, side, dist\
	)\
	_bvhIntersectFirstHit(\
		bvh.position, bvh.index, bvh.bvhBounds, bvh.bvhContents,\
		rayOrigin, rayDirection, faceIndices, faceNormal, barycoord, side, dist\
	)

bool _bvhIntersectFirstHit(
	// bvh info
	sampler2D bvh_position, usampler2D bvh_index, sampler2D bvh_bvhBounds, usampler2D bvh_bvhContents,

	// ray
	vec3 rayOrigin, vec3 rayDirection,

	// output variables split into separate variables due to output precision
	inout uvec4 faceIndices, inout vec3 faceNormal, inout vec3 barycoord,
	inout float side, inout float dist
) {

	// stack needs to be twice as long as the deepest tree we expect because
	// we push both the left and right child onto the stack every traversal
	int pointer = 0;
	uint stack[ BVH_STACK_DEPTH ];
	stack[ 0 ] = 0u;

	float triangleDistance = INFINITY;
	bool found = false;
	while ( pointer > - 1 && pointer < BVH_STACK_DEPTH ) {

		uint currNodeIndex = stack[ pointer ];
		pointer --;

		// check if we intersect the current bounds
		float boundsHitDistance;
		if (
			! intersectsBVHNodeBounds( rayOrigin, rayDirection, bvh_bvhBounds, currNodeIndex, boundsHitDistance )
			|| boundsHitDistance > triangleDistance
		) {

			continue;

		}

		uvec2 boundsInfo = uTexelFetch1D( bvh_bvhContents, currNodeIndex ).xy;
		bool isLeaf = bool( boundsInfo.x & 0xffff0000u );

		if ( isLeaf ) {

			uint count = boundsInfo.x & 0x0000ffffu;
			uint offset = boundsInfo.y;

			found = intersectTriangles(
				bvh_position, bvh_index, offset, count,
				rayOrigin, rayDirection, triangleDistance,
				faceIndices, faceNormal, barycoord, side, dist
			) || found;

		} else {

			uint leftIndex = currNodeIndex + 1u;
			uint splitAxis = boundsInfo.x & 0x0000ffffu;
			uint rightIndex = currNodeIndex + boundsInfo.y;

			bool leftToRight = rayDirection[ splitAxis ] >= 0.0;
			uint c1 = leftToRight ? leftIndex : rightIndex;
			uint c2 = leftToRight ? rightIndex : leftIndex;

			// set c2 in the stack so we traverse it later. We need to keep track of a pointer in
			// the stack while we traverse. The second pointer added is the one that will be
			// traversed first
			pointer ++;
			stack[ pointer ] = c2;

			pointer ++;
			stack[ pointer ] = c1;

		}

	}

	return found;

}
`;

// Note that a struct cannot be used for the hit record including faceIndices, faceNormal, barycoord,
// side, and dist because on some mobile GPUS (such as Adreno) numbers are afforded less precision specifically
// when in a struct leading to inaccurate hit results. See KhronosGroup/WebGL#3351 for more details.

/**
 * Set of shader structs and defined constants used for interacting with the packed BVH in a
 * shader. See [src/webgl/glsl/bvh_struct_definitions.glsl.js](https://github.com/gkjohnson/three-mesh-bvh/blob/master/src/webgl/glsl/bvh_struct_definitions.glsl.js)
 * for full implementations and declarations.
 *
 * Accessed as `BVHShaderGLSL.bvh_struct_definitions`.
 *
 * @section Shader and Texture Packing API
 * @type {string}
 */
const bvh_struct_definitions = /* glsl */`
struct BVH {

	usampler2D index;
	sampler2D position;

	sampler2D bvhBounds;
	usampler2D bvhContents;

};
`;

var BVHShaderGLSL = /*#__PURE__*/Object.freeze({
	__proto__: null,
	bvh_distance_functions: bvh_distance_functions,
	bvh_ray_functions: bvh_ray_functions,
	bvh_struct_definitions: bvh_struct_definitions,
	common_functions: common_functions
});

const shaderStructs = bvh_struct_definitions;
const shaderDistanceFunction = bvh_distance_functions;
const shaderIntersectFunction = `
	${ common_functions }
	${ bvh_ray_functions }
`;

export { AVERAGE, BVH, BVHHelper, BVHShaderGLSL, CENTER, CONTAINED, ExtendedTriangle, FloatVertexAttributeTexture, GeometryBVH, INTERSECTED, IntVertexAttributeTexture, LineBVH, LineLoopBVH, LineSegmentsBVH, MeshBVH, MeshBVHHelper, MeshBVHUniformStruct, NOT_INTERSECTED, ObjectBVH, OrientedBox, PointsBVH, SAH, SKIP_GENERATION, SkinnedMeshBVH, StaticGeometryGenerator, UIntVertexAttributeTexture, VertexAttributeTexture, acceleratedRaycast, computeBatchedBoundsTree, computeBoundsTree, disposeBatchedBoundsTree, disposeBoundsTree, estimateMemoryInBytes, generateIndirectBuffer, getBVHExtremes, getJSONStructure, getTriangleHitPointInfo, shaderDistanceFunction, shaderIntersectFunction, shaderStructs, validateBounds };
//# sourceMappingURL=index.module.js.map
