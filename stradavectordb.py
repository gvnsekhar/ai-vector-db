import json
import openai
import time
from initialize_db import initialize_db
from pymilvus import connections, Collection, utility

import embedding_models.ada_embedding
logs=[]
def current_time_inms():
    obj = time.gmtime(0)
    epoch = time.asctime(obj)
    print("The epoch is:",epoch)
    curr_time = round(time.time()*1000)
    return curr_time
def get_embedding_function(embedding_model):
    if embedding_model == 'text-embedding-ada-002':
        return embedding_models.ada_embedding.get_embedding
def compute_processing_times(start_time,end_time,request):
    processing_times={}
    timings={}
    if "processingTimings" in request:
      processing_times=request["processingTimings"]    
    timings['received_at']=start_time
    timings['finished_at']=end_time
    timings['processing_time']=str(((end_time-start_time)/1000))+" secs"
    processing_times['vector_search']=timings
    return processing_times

def start(event, context):
    
    request = json.loads(event['body'])
    response=start_process(request)
    return response
    
   
def process(input_data, orch_config, block_details,application,milvus):
    response = {}
    response['application_id'] = application['appId']
    response['sending_service'] = 'vector_db'
    response['request'] = input_data
    response['vector_db_results'] = []
    vector_db=orch_config['Applicationlogic']['vectordb']['vectordbList']
    for item in vector_db:
        embed_fields=[]
        fields=[]
        milvus_db_fields=[]
        if item['blockinfo']['blockid'] == block_details['Block_id']:
            print('Block matched')
            result={
                "Block_Name":'vectordb',
                "Block_id":item['blockinfo']['blockid']
            } 
            attributes=item['attributes']
            
            for attribute in attributes:
                if attribute['iskey']==True:
                    embed_fields.append(attribute['logAttributeName'])
                    fl={"source":attribute['source'],"attribute":attribute['logAttributeName']}
                    milvus_db_fields.append(attribute['logAttributeName'])
                    fields.append(fl)

                if attribute['iskey']!=True and attribute['storeinCollection']==True:
                    fl={"source":attribute['source'],"attribute":attribute['logAttributeName']}
                    fields.append(fl)
                    milvus_db_fields.append(attribute['logAttributeName'])

            collection_name=item['collectioninfo']['collectionName']                  
            collection=initialize_db(milvus,milvus_db_fields,collection_name,embed_fields)

            db_data = []
            embeds =[]
            embed_field = get_embedding_function('text-embedding-ada-002')
            
            for f in fields:                
                if f["source"]=="LLM":
                    llm_results=input_data["LLM"]
                    db_data.append([llm_results[0][f["attribute"]]])
                else : 
                    db_data.append([input_data[f["attribute"]]])

            for f in embed_fields:
                embeds.append(embed_field(input_data[f]))

            
            print("db_data\n\n\n\n")
            print(json.dumps(db_data))
             
            vector_data = sum(embeds, [])
            db_data.append([vector_data])
            collection.insert(db_data)
             
    return response

def start_process(request):
    logs.clear()
    start_time=current_time_inms()
    orch_config=request['orchconfig']
    input_data=request['request']
    block_details=request['blockdetails']
    application=request['application']
    milvus=request['milvusconfig']
    openai.api_key=request['openai']['secretkey']
    response=process(input_data, orch_config, block_details,application,milvus) 
    end_time=current_time_inms()
    response['processingTimings']=compute_processing_times(start_time,end_time,request)
    return json.dumps(response)

def main():
    start_time=current_time_inms()
    logs.clear()
    f= open("request.json")
    request=json.load(f)
    f.close()
    response=start_process(request)
    
    print(json.dumps(response))

if __name__ == "__main__":
    main()