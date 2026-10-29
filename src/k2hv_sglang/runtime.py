PACKAGE = __package__
SGLANG_REVISION = "b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf"
MODEL_ARCHITECTURE = "K2HorizonVForConditionalGeneration"
REASONING_PARSER = "k2_horizon"
REGISTRATION_ENVIRONMENT = {
    "SGLANG_EXTERNAL_MODEL_PACKAGE": PACKAGE,
    "SGLANG_EXTERNAL_MM_MODEL_ARCH": MODEL_ARCHITECTURE,
    "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": PACKAGE,
}


SOURCE_SHA256 = {
    "managers/tokenizer_manager.py": "8f29b58741e2cf72bef32eb987865dc020ccc4cfaaa30b4cc0d1019c41782865",
    "layers/attention/torch_native_backend.py": "5da07edadf56f4f4e4b8319ed48216b5a8ddb989224d3e95011b76800bf9926b",
    "batch_invariant_ops/batch_invariant_ops.py": "116abe217cae8d402783006ab0ddf60d9eef9e39a18ebe656d391fe6f5520ec5",
    "layers/conv.py": "f29a48009e55941c23612d3cf3be2114772d56c052fba70f812b06fe6b61d03f",
    "layers/rotary_embedding/utils.py": "828a50285218e40c3429d2f4e75b1d051e3a89d831a7ab005e8c2609278aeddc",
    "models/glm5_next.py": "9ec2d17f12345bbf05cba4f7e9e1b61ac9b952c58f0e7c7b1029b22f8578a1d6",
    "models/xllm.py": "f2aa14ea3af966915bbe4052b90c9c3affed5848ba6bd6b585dd85a3348ae0ab",
    "models/registry.py": "2394ed48eddaf315cfe8e9220969e8ebf796aebd5adb2b7394baa951f6f6e020",
    "multimodal/processors/base_processor.py": "c454d2123b4bf9c865480aad08770936016e0479910fd86754892b41ad1b9d99",
    "managers/mm_utils.py": "97e50fb1c38e374a21e9e092ffa7ebc1301247b43d7413498d7ca271a5178d71",
    "managers/schedule_batch.py": "49d63e0ea2f6982fd6e45a64eeccad06837556c73e53ac75bb87bfdaf521c70c",
    "managers/multimodal_processor.py": "572ff3ab34a8c63665e0ad1504f51c35c3faca83eca9b945583c71a0e5e3d8bc",
    "managers/mm_schedule.py": "817444fee604fb19a009485effaba37566c3816aeeda5a73381f9e5cf97e2673",
    "parser/reasoning_parser.py": "55360b73acfaac21db2ff4644bfc38ff6d744eb4371b15581a2b08e84673dc5e",
    "layers/attention/vision.py": "50472d4dab97bee8ff4ed2dbea3034acfed17d564d2bc56327591e954fe9e20e",
    "managers/tokenizer_control_mixin.py": "9ad9a3fb040bbfe0e9a700e1276c6aa7311905f759f5ff79cb6c39337a2e2da6",
    "managers/scheduler_components/weight_updater.py": "19fdcc5b77a70ede2f95e7ac6835eda9243abe20b165aede22f970e3912950a4",
    "utils/common.py": "1dba3d584ee0f8be088476bedf8274120d09f59cc2019a1a4e2f0f3b5c3903f9",
    "entrypoints/http_server.py": "ee00b694ba319e0332eca03aa265e6c22b7833c2ef145942dfbe5e98b175102e",
}
