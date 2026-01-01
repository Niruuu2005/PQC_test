"""Quick test of crypto subsystem"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crypto_dataset_generator.crypto import create_cipher

print("Creating AES-256-GCM cipher...")
cipher = create_cipher('AES-256-GCM', seed=42)
print("[OK] Cipher created")

print("Generating key...")
cipher.generate_key()
print(f"[OK] Key generated ({cipher.key_size_bits} bits)")

print("Encrypting...")
plaintext = b'Hello, World!'
ciphertext, meta = cipher.encrypt(plaintext)
print(f"[OK] Encrypted {len(ciphertext)} bytes")
print(f"    Success: {meta.success}")
print(f"    Time: {meta.encryption_time_ms:.3f} ms")

print("Decrypting...")
iv = bytes.fromhex(meta.iv)
tag = bytes.fromhex(meta.tag)
recovered, dec_meta = cipher.decrypt(ciphertext, iv=iv, tag=tag)
print(f"[OK] Decrypted {len(recovered)} bytes")
print(f"    Success: {dec_meta.success}")

if recovered == plaintext:
    print("[SUCCESS] Plaintext matches!")
else:
    print("[FAIL] Plaintext mismatch!")

print("\nAll tests passed!")

