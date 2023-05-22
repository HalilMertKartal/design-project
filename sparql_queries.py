from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

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


SELECT DISTINCT ?alternative WHERE { 
  ?variable dct:subject ?concept .
  ?alternative dct:subject ?concept .
  FILTER(regex(?concept, "git", "i"))
}

"""
)
try:
    ret = sparql.queryAndConvert()

    for r in ret["results"]["bindings"]:
        print(r)
except Exception as e:
    print(e)