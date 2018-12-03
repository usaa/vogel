import pandas as pd
import sklearn.pipeline as pipeline
import copy


def make_pipeline(*steps, **kwargs):
    """
    scikit-learn make_pipeline with get_feature_names support
    """
    pl = pipeline.make_pipeline(*steps, **kwargs)
    pl.get_feature_names = steps[-1].get_feature_names
    return pl


def find_label_dicts(pipeline, cat_dicts={}, model_inputs=None):
    """ 
    Find all LabelEncoder's cat_dicts in a pipeline
    """
    out_dict = {}
    for step in pipeline.steps:
        class_name = step[1].__class__.__name__
        if class_name == 'LabelEncoder':
            temp_dict = step[1].cat_dict
            for key, val_dict in temp_dict.items():
                temp_dict[key]['sep'] = step[1].sep
            cat_dicts.update(temp_dict)
        elif class_name == 'FeatureUnion':
            trans_list = step[1].transformer_list
            for trans in trans_list:
                find_label_dicts(trans[1], cat_dicts, model_inputs)
                
    if model_inputs is None:
        return cat_dicts
    else:
        # remove items that will not be in the model
        out_dict = copy.deepcopy(cat_dicts)
        for feature, feature_dict in cat_dicts.items():
            cols = []
            for item in feature_dict['items']:
                if feature_dict['sep'] != '()':
                    cols.append("{0}{1}{2}".format(feature, feature_dict['sep'], item))
                else:
                    cols.append("{0} ({1})".format(feature, item))

            if len(set(cols) & set(model_inputs)) == 0:
                del out_dict[feature]
        return out_dict 

def label_dict_formater(sep, feature, value):
    if sep != '()':
        return "{0}{1}{2}".format(feature, sep, value)
    else:
        return "{0} ({1})".format(feature, value)