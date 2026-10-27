SGLANG_REVISION = "b8d46f2f309fdada4bbb05ef4e90ac4adf7a3bcf"
MODEL_ARCHITECTURE = "K2HorizonVForConditionalGeneration"
REASONING_PARSER = "k2_horizon"


def registration_environment():
    return {
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "k2hv_sglang",
        "SGLANG_EXTERNAL_MM_MODEL_ARCH": MODEL_ARCHITECTURE,
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "k2hv_sglang",
    }


SOURCE_SHA256 = {
    "models/glm5_next.py": "9ec2d17f12345bbf05cba4f7e9e1b61ac9b952c58f0e7c7b1029b22f8578a1d6",
    "models/glm_ocr.py": "9acab22342f46d398989fb7dfff2b242659ac87eef13194202ee5c1bad3f0495",
    "models/glm4v.py": "675578397b855a27f775acf727892ae875d0df60b555f3f97523ec7b060f61f0",
    "models/xllm.py": "f2aa14ea3af966915bbe4052b90c9c3affed5848ba6bd6b585dd85a3348ae0ab",
    "models/registry.py": "2394ed48eddaf315cfe8e9220969e8ebf796aebd5adb2b7394baa951f6f6e020",
    "multimodal/processors/base_processor.py": "c454d2123b4bf9c865480aad08770936016e0479910fd86754892b41ad1b9d99",
    "managers/mm_utils.py": "97e50fb1c38e374a21e9e092ffa7ebc1301247b43d7413498d7ca271a5178d71",
    "managers/schedule_batch.py": "49d63e0ea2f6982fd6e45a64eeccad06837556c73e53ac75bb87bfdaf521c70c",
    "managers/multimodal_processor.py": "572ff3ab34a8c63665e0ad1504f51c35c3faca83eca9b945583c71a0e5e3d8bc",
    "managers/mm_schedule.py": "817444fee604fb19a009485effaba37566c3816aeeda5a73381f9e5cf97e2673",
    "mem_cache/multimodal_cache.py": "26dbec397b32f80e96eb1631634fab5a6297f551c0e423f3858085fc0fc186b4",
    "parser/reasoning_parser.py": "55360b73acfaac21db2ff4644bfc38ff6d744eb4371b15581a2b08e84673dc5e",
    "parser/reasoning_parser_names.py": "892e7f03a3805b3051f66756b081055467b8dc2369d90f60c43b3b1ef1d6b09f",
    "layers/attention/vision.py": "50472d4dab97bee8ff4ed2dbea3034acfed17d564d2bc56327591e954fe9e20e",
    "layers/conv.py": "f29a48009e55941c23612d3cf3be2114772d56c052fba70f812b06fe6b61d03f",
    "configs/model_config.py": "a7354376a52ed38dc818fb4cae7b1544127f5a44397b75554c3d9049ca07af28",
    "managers/tokenizer_manager.py": "fd516b589b3caf1cbc7e09c14af3674ecdea2f399844d41f3a1574716ab83fdb"
}
