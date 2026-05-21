import json

from openai import OpenAI


def get_current_temperature(location: str, unit: str = 'celsius'):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        'temperature': 26.1,
        'location': location,
        'unit': unit,
    }


def get_temperature_date(location: str, date: str, unit: str = 'celsius'):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        'temperature': 25.9,
        'location': location,
        'date': date,
        'unit': unit,
    }


def get_function_by_name(name):
    if name == 'get_current_temperature':
        return get_current_temperature
    if name == 'get_temperature_date':
        return get_temperature_date


tools = [{
    'type': 'function',
    'function': {
        'name': 'get_current_temperature',
        'description': 'Get current temperature at a location.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit'],
                    'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                }
            },
            'required': ['location']
        }
    }
}, {
    'type': 'function',
    'function': {
        'name': 'get_temperature_date',
        'description': 'Get temperature at a location and date.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
                },
                'date': {
                    'type': 'string',
                    'description': 'The date to get the temperature for, in the format \'Year-Month-Day\'.'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit'],
                    'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                }
            },
            'required': ['location', 'date']
        }
    }
}]

messages = [{
    'role': 'user',
    'content': 'Today is 2024-11-14, What\'s the temperature in San Francisco now? How about tomorrow?'
}]

openai_api_key = 'EMPTY'
openai_api_base = 'http://0.0.0.0:23333/v1'
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = client.models.list().data[0].id
response = client.chat.completions.create(model=model_name,
                                          messages=messages,
                                          max_tokens=32768,
                                          temperature=0.8,
                                          top_p=0.8,
                                          stream=False,
                                          extra_body=dict(spaces_between_special_tokens=False, enable_thinking=False),
                                          tools=tools)
print(response.choices[0].message)
messages.append(response.choices[0].message)

for tool_call in response.choices[0].message.tool_calls:
    tool_call_args = json.loads(tool_call.function.arguments)
    tool_call_result = get_function_by_name(tool_call.function.name)(**tool_call_args)
    tool_call_result = json.dumps(tool_call_result, ensure_ascii=False)
    messages.append({
        'role': 'tool',
        'name': tool_call.function.name,
        'content': tool_call_result,
        'tool_call_id': tool_call.id
    })

response = client.chat.completions.create(model=model_name,
                                          messages=messages,
                                          temperature=0.8,
                                          top_p=0.8,
                                          stream=False,
                                          extra_body=dict(spaces_between_special_tokens=False, enable_thinking=False),
                                          tools=tools)
print(response.choices[0].message.content)
