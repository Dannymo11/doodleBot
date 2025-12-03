# The goal of this script is to be able to refine one sketch, minimizing the combined semantic and style loss

# 1. Embedding Alignment

# represent the sketch as a differntiable set of stroke parameters

# compute CLIP image embedding
# compute CLIP text embedding
# calculate cosine similarity
# calculate semantic loss (L = 1 - cos_similarity)



# 2. Style Preservation

# 3. Joint Optimization Loop