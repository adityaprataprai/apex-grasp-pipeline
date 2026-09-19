from apex_grasp.stage1_language.parser import RuleParser

def test_simple_rule_parser():
    ont={'objects':['bottle','carton'],'attributes':{'color':['blue'],'condition':[],'size':[],'transparency':[]}}
    p=RuleParser('test',ont).parse('Grab the blue bottle.')
    assert p['action']=='GRASP' and p['target']['category']=='bottle' and p['target']['attributes']['color']=='blue'
