import collections

EmbeddingCache = collections.namedtuple( 'EmbeddingCache', 'embeddingId, firstBinId, type, features, ttf' )
EmbeddingCacheTest = collections.namedtuple( 'EmbeddingCacheTest', 'embeddingId, type, features' )