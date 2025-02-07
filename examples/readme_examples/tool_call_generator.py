from synth_gen.generation.json_generator import ToolCallGenerator

generator = ToolCallGenerator()
generation = generator.generate_training_data()
generation = generation[0]
print("TOOL_NAME:", generation.tool_name)
print("TOOL_DESCRIPTION:", generation.tool_description)
print("TOOL_PARAMETERS:", generation.tool_parameters)
print("PROMPT:", generation.prompt.messages[0].content)
print("GENERATED_TOOL_CALL:", generation.response.content)
