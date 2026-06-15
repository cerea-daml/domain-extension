#Pipeline

import logging

from noether.core.schemas.dataset import PipelineConfig

from noether.data.pipeline import MultiStagePipeline
from noether.data.pipeline.sample_processors import ConcatTensorSampleProcessor
from noether.data.pipeline.collators import DefaultCollator

from academic_cases.datasets import AcademicDataSpecs

logger = logging.getLogger(__name__)

class AcademicCasesPipelineConfig(PipelineConfig):
    kind: str
    data_specs: AcademicDataSpecs
    
    model_config = {
        "extra": "forbid",}

class AcademicCasesPipeline(MultiStagePipeline):
    '''Pipeline for simple academic datasets. Simply collate input and target feature tensors'''
    def __init__(
        self,
            config: AcademicCasesPipelineConfig
    ):
        
        sample_processors = [
            ConcatTensorSampleProcessor(
                config.data_specs.output_targets.keys(),
                target_key = 'target_feature',
                dim = -1
            )
        ]
             
        sample_processors.append(
            ConcatTensorSampleProcessor(
                config.data_specs.input_features.keys(),
                target_key = 'input_feature',
                dim = -1
            )
        )

        collators = [
            DefaultCollator(items=["position", "input_feature", "target_feature"],
                            optional_items=['index']),
        ]

        super().__init__(

            sample_processors=sample_processors,
            collators=collators
        )