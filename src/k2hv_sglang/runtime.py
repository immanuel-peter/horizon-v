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
}
