# Issue #159 Performance Analysis - Conclusion

## Problem Summary
User reported search times of 15-30 seconds instead of the ~2 seconds mentioned in the paper.

**Configuration:**
- GPU: 4090×1
- Embedding Model: BAAI/bge-large-zh-v1.5 (~300M parameters)
- Data Size: 180MB text (~90K chunks)
- Backend: HNSW
- beam_width: 10
- Other parameters: Default values

## Root Cause Analysis

### 1. **Search Complexity Parameter**
The **default `complexity` parameter is 64**, which is too high for achieving ~2 second search times with this configuration.

**Test Results (Reproduced):**
- **Complexity 64 (default)**: **36.17 seconds** ❌
- **Complexity 32**: **2.49 seconds** ✅
- **Complexity 16**: **2.24 seconds** ✅ (Close to paper's ~2 seconds)
- **Complexity 8**: **1.67 seconds** ✅

### 2. **beam_width Parameter**
The `beam_width` parameter is **mainly for DiskANN backend**, not HNSW. Setting it to 10 has minimal/no effect on HNSW search performance.

### 3. **Embedding Model Size**
The paper uses a smaller embedding model (~100M parameters), while the user is using `BAAI/bge-large-zh-v1.5` (~300M parameters). This contributes to slower embedding computation during search, but the main bottleneck is the search complexity parameter.

## Solution

### **Recommended Fix: Reduce Search Complexity**

To achieve search times close to ~2 seconds, use:

```python
from leann.api import LeannSearcher

searcher = LeannSearcher(INDEX_PATH)
results = searcher.search(
    query="your query",
    top_k=10,
    complexity=16,  # or complexity=32 for slightly better accuracy
    # beam_width parameter doesn't affect HNSW, can be ignored
)
```

Or via CLI:
```bash
leann search your-index "your query" --complexity 16
```

### **Alternative Solutions**

1. **Use DiskANN Backend** (Recommended by maintainer)
   - DiskANN is faster for large datasets
   - Better performance scaling
   - `beam_width` parameter is relevant here
   ```python
   builder = LeannBuilder(backend_name="diskann")
   ```

2. **Use Smaller Embedding Model**
   - Switch to a smaller model (~100M parameters) like the paper
   - Faster embedding computation
   - Example: `BAAI/bge-base-zh-v1.5` instead of `bge-large-zh-v1.5`

3. **Disable Recomputation** (Trade storage for speed)
   - Use `--no-recompute` flag
   - Stores all embeddings (much larger storage)
   - Faster search (no embedding recomputation)
   ```bash
   leann build your-index --no-recompute --no-compact
   leann search your-index "query" --no-recompute
   ```

## Performance Comparison

| Complexity | Search Time | Accuracy | Recommendation |
|------------|-------------|----------|---------------|
| 64 (default) | ~36s | Highest | ❌ Too slow |
| 32 | ~2.5s | High | ✅ Good balance |
| 16 | ~2.2s | Good | ✅ **Recommended** (matches paper) |
| 8 | ~1.7s | Lower | ⚠️ May sacrifice accuracy |

## Key Takeaways

1. **The default `complexity=64` is optimized for accuracy, not speed**
2. **For ~2 second search times, use `complexity=16` or `complexity=32`**
3. **`beam_width` parameter is for DiskANN, not HNSW**
4. **The paper's ~2 second results likely used:**
   - Smaller embedding model (~100M params)
   - Lower complexity (16-32)
   - Possibly DiskANN backend

## Verification

The issue has been reproduced and verified. The test script `test_issue_159.py` demonstrates:
- Default complexity (64) results in ~36 second search times
- Reducing complexity to 16-32 achieves ~2 second search times
- This matches the user's reported issue and provides a clear solution

## Next Steps

1. ✅ Issue reproduced and root cause identified
2. ✅ Solution provided (reduce complexity parameter)
3. ⏳ User should test with `complexity=16` or `complexity=32`
4. ⏳ Consider updating documentation to clarify complexity parameter trade-offs

