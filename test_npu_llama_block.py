
import torch
import intel_npu_acceleration.compiler as npu_compiler
from tiny_llama import Llama, LlamaConfig
import time

def test_npu_llama():
    # 1. Config & Model
    conf = LlamaConfig(
        dim=64,
        n_layers=1, # Start with 1 layer to debug
        n_heads=4,
        n_kv_heads=4,
        vocab_size=128,
        max_seq_len=32
    )
    model = Llama(conf)
    model.eval()

    # 2. Inputs
    # We compile the Transformer Block, not the whole Llama model because Embedding/Output are simple.
    # Actually, let's try compiling the Block.
    
    block = model.layers[0]
    
    # Input tensor 'h' shape (1, seqlen, dim)
    seqlen = 10
    h = torch.randn(1, seqlen, conf.dim)
    start_pos = 0
    
    # Freqs cis (now cos/sin)
    freqs_cos = model.freqs_cos[start_pos : start_pos + seqlen]
    freqs_sin = model.freqs_sin[start_pos : start_pos + seqlen]
    
    # Mask (causal)
    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"))
    mask = torch.triu(mask, diagonal=start_pos + 1)
    
    # 3. Compile
    print("Compiling Transformer Block to NPU...")
    try:
        npu_block = npu_compiler.compile_to_npu(block, (h, start_pos, freqs_cos, freqs_sin, mask))
        print("Compilation successful.")
        
        # Run
        print("Running Inference...")
        start = time.time()
        out_npu = npu_block(h, start_pos, freqs_cos, freqs_sin, mask)
        print(f"NPU Time: {time.time() - start:.4f}s")
        
        out_cpu = block(h, start_pos, freqs_cos, freqs_sin, mask)
        
        if torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2):
            print("Verification SUCCESS")
        else:
            print("Verification FAILED")
            print(f"Diff: {(out_npu - out_cpu).abs().max().item()}")
            
    except Exception as e:
        print(f"Compilation/Run Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_npu_llama()
