CUDA_VISIBLE_DEVICES=1,2 lmdeploy serve api_server $model_path \
    --reasoning-parser intern-s1 \
    --tool-call-parser intern-s1 \
    --model-name Intern-S1-mini \
    --backend pytorch \
    --server-port 8002 \
    --tp 2



curl --location 'http://0.0.0.0:8002/v1/chat/completions' \
--header 'Authorization: xxx' \
--header 'Content-Type: application/json' \
--data '{
  "model": "Intern-S1-mini",
 "spaces_between_special_tokens": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "mkZ9BQ",
        "parameters": {
          "type": "object",
          "required": [],
          "properties": {}
        },
        "description": "获取用户当前时区的时间。"
      }
    }
  ],
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": "几点啦"
    },
    {
      "role": "assistant",
      "content": "null"
    },
    {
      "role": "user",
      "content": "几点啦"
    },
    {
      "role": "assistant",
      "content": "null"
    },
    {
      "role": "user",
      "content": "几点啦"
    },
    {
      "role": "assistant",
      "content": "<think>\n好的，用户连续三次问“几点啦”，我需要调用工具来获取当前时间。首先，检查可用的工具，发现有一个名为l2dVFb的函数，它不需要任何参数，可以返回用户时区的时间。由于用户多次询问，可能希望得到准确的当前时间，所以应该调用这个工具。每次询问都触发一次调用，确保返回的时间是最新的。现在生成对应的工具调用JSON，不带参数，因为函数不需要任何参数。确认工具调用的格式正确，放入XML标签中。完成。\n</think>\n\n<think>\n好的，用户问了三次“几点啦”，我之前调用了工具来获取时间，现在工具返回了结果。需要确认返回的时间格式是否正确，然后回复用户。用户可能只是想知道自己所在时区的当前时间，所以直接给出具体的时间和星期几应该就可以了。检查一下返回的时间是否合理，比如年份是2025年，月份和日期是否正确。然后组织语言，用友好的方式告诉用户当前的时间，比如“现在是2025年10月10日18点21分29秒，星期五”。确保回复简洁明了，符合用户的需求。\n</think>\n\n现在是2025年10月10日18点21分29秒，星期五。"
    },
    {
      "role": "user",
      "content": "几点啦"
    }
  ],
  "max_tokens": 10000,
  "tool_choice": "auto"
}'



curl --location 'http://0.0.0.0:8002/v1/chat/completions' \
--header 'Authorization: xxx' \
--header 'Content-Type: application/json' \
--data '{
  "model": "Intern-S1-mini",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_temperature",
        "parameters": {
          "type": "object",
          "required": [
            "location"
          ],
          "properties": {
            "unit": {
              "enum": [
                "celsius",
                "fahrenheit"
              ],
              "type": "string",
              "description": "The unit to return the temperature in. Defaults to \"celsius\"."
            },
            "location": {
              "type": "string",
              "description": "The location to get the temperature for, in the format \"City, State, Country\"."
            }
          }
        },
        "description": "Get current temperature at a location."
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_temperature_date",
        "parameters": {
          "type": "object",
          "required": [
            "location",
            "date"
          ],
          "properties": {
            "date": {
              "type": "string",
              "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."
            },
            "unit": {
              "enum": [
                "celsius",
                "fahrenheit"
              ],
              "type": "string",
              "description": "The unit to return the temperature in. Defaults to \"celsius\"."
            },
            "location": {
              "type": "string",
              "description": "The location to get the temperature for, in the format \"City, State, Country\"."
            }
          }
        },
        "description": "Get temperature at a location and date."
      }
    }
  ],
  "top_p": 0.8,
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": "Today is 2024-11-14, What''s the temperature in San Francisco now? How about tomorrow?"
    }
  ],
  "max_tokens": 32768,
  "temperature": 0.8,
  "enable_thinking": false,
  "spaces_between_special_tokens": false
}'
