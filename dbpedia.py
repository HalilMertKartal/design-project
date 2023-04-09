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

def exportAttributes(csv_path):
    df = pd.read_csv(csv_path)
    allTypes = {}
    allEntities = {}
    ct = 0
    for i in df["text"]:
        try:
            allTypes[ct] = [_ for _ in return_types(i)][0]
            allEntities[ct] = [_ for _ in return_entities(i)][0]
            ct += 1
        except:
            print("error")
        

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

    with open("allEntities.json", "w") as outfile:
        json.dump(allEntities, outfile)

if __name__ == "__main__":
    main()
