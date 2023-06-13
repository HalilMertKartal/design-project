from SPARQLWrapper import SPARQLWrapper, JSON
import json
import re

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

resultDict = {}

def queryMaker(wordToQuery):

  sparql.setReturnFormat(JSON)
  sparql.setQuery(
  """
  PREFIX      owl: <http://www.w3.org/2002/07/owl#>
  PREFIX      xsd: <http://www.w3.org/2001/XMLSchema#>
  PREFIX     rdfs: <http://www.w3.org/2000/01/rdf-schema#>
  PREFIX      rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
  PREFIX     foaf: <http://xmlns.com/foaf/0.1/>
  PREFIX       dc: <http://purl.org/dc/elements/1.1/>
  PREFIX      res: <http://dbpedia.org/resource/>
  PREFIX dbpedia2: <http://dbpedia.org/property/>
  PREFIX  dbpedia: <http://dbpedia.org/>
  PREFIX     skos: <http://www.w3.org/2004/02/skos/core#>
  PREFIX dbc:	<http://dbpedia.org/resource/Category:>
  PREFIX dbo:	<http://dbpedia.org/ontology/>
  PREFIX dbp:	<http://dbpedia.org/property/>
  PREFIX ns: <http://example.org/namespace/>
  PREFIX dct: <http://purl.org/dc/terms/>


  SELECT DISTINCT ?software WHERE {
    {
      ?software dbo:genre ?genre ;
                rdf:type dbo:Software .
      ?genre rdfs:label ?genreLabel .
      FILTER (lcase(str(?genreLabel)) = lcase("%s"))
    }
    UNION
    {
      ?software dct:subject ?concept ;
                rdf:type dbo:Software .
      ?concept rdfs:label ?conceptLabel .
      FILTER (lcase(str(?conceptLabel)) = lcase("%s"))
    }
    

  }
  """
  %(wordToQuery, wordToQuery)
  )
  try:
      ret = sparql.queryAndConvert()

      for r in ret["results"]["bindings"]:
          for i in r:
            for j in r[i]:
              if j == "value":
                result = re.search(r'/([^/]+)$', r[i][j]).group(1)
                return result
  except Exception as e:
      print(e)

with open('allEntities_v2.json') as json_file:
    data = json.load(json_file)
 
    for i in range(1519):
      key = str(i)
      if len(data[key]) == 0:
        resultDict[key] = []
      else:
        resultArr = []
        for j in data[key]:
          result = queryMaker(j)
          resultArr.append(result)

        resultDict[key] = resultArr

print(resultDict)
with open("allRecommendations.json", "w") as outfile:
  json.dump(resultDict, outfile)
