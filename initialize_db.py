import json
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

def initialize_db(milvus,fields,collection_name,embed_fields):
     
    hostName=milvus['host']
    portNo=milvus['port']
    milvus_client = connections.connect(
      host=hostName,
      port=portNo
    ) 

    

    if utility.has_collection(collection_name):
        return Collection(name=collection_name)

    index_params = {
        'index_type': 'HNSW',
        'metric_type': 'L2',
        'params': {
            'M': 8,
            'efConstruction': 64
        },
    }

    
    other_fields = [FieldSchema(name=field, dtype=DataType.VARCHAR, max_length=5000) for field in fields[0:]]
    vector_dim = 1536 * len(embed_fields)
    # Using only one vector for now
     
     
    vector_field = FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
    primary_field = FieldSchema(name='id', dtype=DataType.INT64,auto_id=True , is_primary=True)
    schema = CollectionSchema(fields=[primary_field]+other_fields+[vector_field])
    print("Fields\n\n\n")
    
    print([primary_field]+other_fields+[vector_field])
    collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)
    collection.create_index('vector', index_params)
    collection.load()

    return collection
