from orion_kl_ml.models import embed


def test_smiles_batch():
    smiles = ["c1ccccc1", "CC#N", "C#CC#CC#CC#N"]
    embeddings = embed.embed_smiles_batch(smiles)
