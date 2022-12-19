from owlready2 import *

software_world = World()

onto = software_world.get_ontology("https://raw.githubusercontent.com/allysonlister/swo/master/swo.owl").load()

# print(list(onto.classes()))
graph = software_world.as_rdflib_graph()

query_result = list(software_world.sparql("""
           SELECT ?x
           { ?x rdfs:label "software" .
              }
    """)
    )

print(query_result)