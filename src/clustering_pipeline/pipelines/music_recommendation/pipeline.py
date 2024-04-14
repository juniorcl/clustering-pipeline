"""
This is a boilerplate pipeline 'music_recommendation'
generated using Kedro 0.19.3
"""

from . import nodes
from kedro.pipeline import Pipeline, pipeline, node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.preprocessing_data,
                inputs=["music_recommendation_raw_data_data"],
                outputs="music_recommendation_preprocessed_data",
                name="music_recommendation_preprocessing_node"
            ),
            node(
                func=nodes.train_model,
                inputs=["music_recommendation_preprocessed_data"],
                outputs="music_recommendation_pipe_pipeline",
                name="music_recommendation_train_model_node"
            )
        ]
    )
