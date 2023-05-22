import spacy
import pandas as pd
import json

def return_types(decision):
    nlp = spacy.blank('en')
    # nlp.add_pipe('dbpedia_spotlight', config={'types': None, 'confidence': 0.75})
    #nlp.add_pipe('dbpedia_spotlight', config={'process': 'candidates'})
    #nlp.get_pipe('dbpedia_spotlight').types = None
    nlp.add_pipe('dbpedia_spotlight')
    doc = nlp(decision)
    # print([(ent._.dbpedia_raw_result['@types']) for ent in doc.ents])
    # doc = nlp("add python api multilayer perceptron classifier add python api multilayer perceptron classifier")
    # doc.ents
    # print([(ent.text, ent.kb_id_, ent._.dbpedia_raw_result['resource']['@contextualScore']) for ent in doc.ents])
    data = [x._.dbpedia_raw_result['@types'] for x in doc.ents]
    unique_data = set(data)
    data = [[decision, list(unique_data)]]
    my_df = pd.DataFrame(data, columns=["text", "types"])
    return my_df["types"]

def return_entities(decision):
    nlp = spacy.blank('en')
    # nlp.add_pipe('dbpedia_spotlight', config={'types': None, 'confidence': 0.75})
    #nlp.add_pipe('dbpedia_spotlight', config={'process': 'candidates'})
    #nlp.get_pipe('dbpedia_spotlight').types = None
    nlp.add_pipe('dbpedia_spotlight')
    doc = nlp(decision)
    # doc = nlp("add python api multilayer perceptron classifier add python api multilayer perceptron classifier")
    # doc.ents
    # print([(ent.text, ent.kb_id_, ent._.dbpedia_raw_result['resource']['@contextualScore']) for ent in doc.ents])
    data = [x for x in list(str(x).lower() for x in doc.ents)]
    unique_data = set(data)
    data = [[decision, list(unique_data)]]
    my_df = pd.DataFrame(data, columns=["text", "entities"])
    return my_df["entities"]

def methodBOC():
    nlp = spacy.blank('en')
    nlp.add_pipe('dbpedia_spotlight', config={
                 'process': "annotate"})
    return nlp

def methodAT(decision):
    # methodBOC() sonuçlarına typeları ekle
    nlp = methodBOC()
    doc = nlp(decision)
    data = [x._.dbpedia_raw_result['@types'] for x in doc.ents]
    
    data = sorted(data)
    dataWords = []
    unique_data = set(data)
    for i in unique_data:
    # ignore empty strings
        if i != "":
            # Get last word as the type
            # Dbpedia:
            splitted = i.split(":")
            word = str(splitted[-1]).lower()
            dataWords.append(word)
    
    data = [[decision, decision.split(" ") + list(dataWords)]]
    my_df = pd.DataFrame(data, columns=["text", "enrichedTextWithTypes"])
    return my_df["enrichedTextWithTypes"]

def methodATT(decision):
    nlp = methodBOC()
    doc = nlp(decision)
    # Topics yok
    data = [x._.dbpedia_raw_result['@topics'] for x in doc.ents]
    return data

def methodATTC():
    pass

def exportAttributes(csv_path):
    df = pd.read_csv(csv_path)
    allTypes = {}
    allEntities = {}
    ct = 0
    for i in df["text"]:
        if ct == 1482 or ct == 1502 or ct == 1503:
            print(i)
            allEntities[ct] = []
            ct += 1
            continue
        try:
            allTypes[ct] = [_ for _ in return_types(i)][0]
            allEntities[ct] = [_ for _ in return_entities(i)][0]
            ct += 1
        except:
            allEntities[ct] = []
        

    return(allTypes, allEntities)

def main():
    """
    text = 'platform dependent unsafe way verbose rename platform dependent unsafe platform'
    types = return_types(text)
    entities = return_entities(text)
    print([_ for _ in types])
    print([_ for _ in entities])
    """

   
    csv_path = "dataset\design_decisions_v1.csv"
    allTypes, allEntities = exportAttributes(csv_path)
    print(allTypes)
    print(allEntities)
    with open("allTypes.json", "w") as outfile:
        json.dump(allTypes, outfile)

    with open("allEntities_v2.json", "w") as outfile:
        json.dump(allEntities, outfile)
        
    """
    text = "method survived code review since v exposes jbl as types let remove public api expect one calls directly hi deals solve least squares"
    df2 = methodAT(text)
    print(df2[0])

    print(methodATT(text))
     """

if __name__ == "__main__":
    main()
