#!/usr/bin/env node

/**
 * Venice AI MCP Server
 * Provides Claude Code access to Venice AI API - privacy-focused uncensored AI
 *
 * Key Features:
 * - Privacy-first: No data storage or logging
 * - Uncensored models available
 * - Web search integration
 * - Image generation (Nano Banana Pro)
 * - OpenAI-compatible API
 *
 * Models available:
 * - llama-3.3-70b: Default chat model
 * - deepseek-r1-671b: Advanced reasoning (R1)
 * - qwen-2.5-coder-32b: Code specialist
 * - dolphin-2.9.3-mistral-7b: Uncensored chat
 * - flux-dev: Image generation
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import OpenAI from 'openai';

// Initialize Venice client (OpenAI-compatible API)
const venice = new OpenAI({
  apiKey: process.env.VENICE_API_KEY,
  baseURL: 'https://api.venice.ai/api/v1',
});

// Create MCP server
const server = new McpServer({
  name: 'venice',
  version: '1.0.0',
});

// ============================================
// TYPES
// ============================================

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

// ============================================
// CHAT TOOLS
// ============================================

server.tool(
  'venice_chat',
  'Send a chat completion to Venice AI. Privacy-focused with uncensored models available.',
  {
    messages: z.array(z.object({
      role: z.enum(['system', 'user', 'assistant']),
      content: z.string(),
    })).describe('Array of messages in the conversation'),
    model: z.string().optional().describe('Model: llama-3.3-70b (default), deepseek-r1-671b, qwen-2.5-coder-32b, dolphin-2.9.3-mistral-7b'),
    temperature: z.number().optional().describe('Sampling temperature 0-2. Default: 0.7'),
    max_tokens: z.number().optional().describe('Maximum tokens to generate'),
    web_search: z.enum(['on', 'off', 'auto']).optional().describe('Enable web search: on, off, auto'),
  },
  async (params: { messages: ChatMessage[]; model?: string; temperature?: number; max_tokens?: number; web_search?: string }) => {
    try {
      const { messages, model, temperature, max_tokens, web_search } = params;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const requestBody: any = {
        model: model || 'llama-3.3-70b',
        messages,
        temperature: temperature ?? 0.7,
        max_tokens,
      };

      // Add Venice-specific parameters if web search is enabled
      if (web_search) {
        requestBody.venice_parameters = {
          enable_web_search: web_search,
        };
      }

      const response = await venice.chat.completions.create(requestBody);

      const content = response.choices[0]?.message?.content || 'No response';
      const usage = response.usage;

      return {
        content: [
          {
            type: 'text' as const,
            text: `${content}\n\n---\nTokens: ${usage?.prompt_tokens} prompt + ${usage?.completion_tokens} completion = ${usage?.total_tokens} total`,
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice API error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  'venice_complete',
  'Simple one-shot completion with Venice AI.',
  {
    prompt: z.string().describe('The prompt to send'),
    system: z.string().optional().describe('Optional system message'),
    model: z.string().optional().describe('Model to use. Default: llama-3.3-70b'),
    temperature: z.number().optional().describe('Temperature 0-2'),
  },
  async (params: { prompt: string; system?: string; model?: string; temperature?: number }) => {
    try {
      const { prompt, system, model, temperature } = params;
      const messages: ChatMessage[] = [];
      if (system) {
        messages.push({ role: 'system', content: system });
      }
      messages.push({ role: 'user', content: prompt });

      const response = await venice.chat.completions.create({
        model: model || 'llama-3.3-70b',
        messages,
        temperature: temperature ?? 0.7,
      });

      return {
        content: [
          {
            type: 'text' as const,
            text: response.choices[0]?.message?.content || 'No response',
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice API error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ============================================
// UNCENSORED CHAT
// ============================================

server.tool(
  'venice_uncensored',
  'Chat with Venice uncensored model. No content restrictions. Privacy-focused.',
  {
    prompt: z.string().describe('The prompt'),
    system: z.string().optional().describe('Optional system message'),
  },
  async (params: { prompt: string; system?: string }) => {
    try {
      const { prompt, system } = params;
      const messages: ChatMessage[] = [];
      if (system) {
        messages.push({ role: 'system', content: system });
      }
      messages.push({ role: 'user', content: prompt });

      const response = await venice.chat.completions.create({
        model: 'dolphin-2.9.3-mistral-7b',
        messages,
        temperature: 0.8,
      });

      return {
        content: [
          {
            type: 'text' as const,
            text: response.choices[0]?.message?.content || 'No response',
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ============================================
// WEB SEARCH TOOL
// ============================================

server.tool(
  'venice_search',
  'Ask Venice with web search enabled. Gets real-time information from the web.',
  {
    query: z.string().describe('Question or query to search for'),
    model: z.string().optional().describe('Model to use. Default: llama-3.3-70b'),
  },
  async (params: { query: string; model?: string }) => {
    try {
      const { query, model } = params;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const requestBody: any = {
        model: model || 'llama-3.3-70b',
        messages: [
          { role: 'user', content: query },
        ],
        temperature: 0.5,
        venice_parameters: {
          enable_web_search: 'on',
        },
      };

      const response = await venice.chat.completions.create(requestBody);

      return {
        content: [
          {
            type: 'text' as const,
            text: response.choices[0]?.message?.content || 'No results',
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice search error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ============================================
// REASONING TOOL (DeepSeek R1)
// ============================================

server.tool(
  'venice_reason',
  'Use DeepSeek R1 via Venice for complex reasoning. Step-by-step thinking.',
  {
    problem: z.string().describe('The problem or question requiring deep reasoning'),
    context: z.string().optional().describe('Additional context'),
    max_tokens: z.number().optional().describe('Max tokens. Default: 8192'),
  },
  async (params: { problem: string; context?: string; max_tokens?: number }) => {
    try {
      const { problem, context, max_tokens } = params;

      let prompt = problem;
      if (context) {
        prompt = `Context:\n${context}\n\nProblem: ${problem}`;
      }

      const response = await venice.chat.completions.create({
        model: 'deepseek-r1-671b',
        messages: [
          { role: 'system', content: 'Think step by step. Show your reasoning process clearly.' },
          { role: 'user', content: prompt },
        ],
        max_tokens: max_tokens || 8192,
        temperature: 0,
      });

      return {
        content: [
          {
            type: 'text' as const,
            text: response.choices[0]?.message?.content || 'No response',
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice R1 error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ============================================
// CODE TOOLS
// ============================================

server.tool(
  'venice_code',
  'Generate code using Qwen Coder via Venice. Optimized for programming.',
  {
    task: z.string().describe('Description of what code to generate'),
    language: z.string().optional().describe('Programming language'),
    context: z.string().optional().describe('Additional context or existing code'),
  },
  async (params: { task: string; language?: string; context?: string }) => {
    try {
      const { task, language, context } = params;
      const systemPrompt = `You are an expert programmer. Generate clean, efficient, well-documented code.${language ? ` Use ${language}.` : ''} Only output the code with minimal explanation.`;

      let userPrompt = task;
      if (context) {
        userPrompt = `Context:\n${context}\n\nTask: ${task}`;
      }

      const response = await venice.chat.completions.create({
        model: 'qwen-2.5-coder-32b',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.2,
      });

      return {
        content: [
          {
            type: 'text' as const,
            text: response.choices[0]?.message?.content || 'No code generated',
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice Coder error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  'venice_analyze',
  'Analyze code using Venice.',
  {
    code: z.string().describe('The code to analyze'),
    task: z.enum(['explain', 'review', 'bugs', 'improve', 'security']).describe('Type of analysis'),
  },
  async (params: { code: string; task: string }) => {
    try {
      const { code, task } = params;
      const taskPrompts: Record<string, string> = {
        explain: 'Explain what this code does in detail.',
        review: 'Review this code for quality and best practices.',
        bugs: 'Find any bugs or potential issues.',
        improve: 'Suggest improvements.',
        security: 'Analyze for security vulnerabilities.',
      };

      const response = await venice.chat.completions.create({
        model: 'qwen-2.5-coder-32b',
        messages: [
          { role: 'user', content: `${taskPrompts[task]}\n\nCode:\n\`\`\`\n${code}\n\`\`\`` },
        ],
        temperature: 0.3,
      });

      return {
        content: [
          {
            type: 'text' as const,
            text: response.choices[0]?.message?.content || 'No analysis',
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ============================================
// IMAGE GENERATION
// ============================================

server.tool(
  'venice_image',
  'Generate images using Venice AI (Flux model). Returns image URL.',
  {
    prompt: z.string().describe('Image description/prompt'),
    style: z.string().optional().describe('Style hint (e.g., "photorealistic", "anime", "oil painting")'),
    negative_prompt: z.string().optional().describe('What to avoid in the image'),
    width: z.number().optional().describe('Image width (default 1024)'),
    height: z.number().optional().describe('Image height (default 1024)'),
  },
  async (params: { prompt: string; style?: string; negative_prompt?: string; width?: number; height?: number }) => {
    try {
      const { prompt, style, negative_prompt, width, height } = params;

      let fullPrompt = prompt;
      if (style) {
        fullPrompt = `${style} style: ${prompt}`;
      }

      // Use OpenAI-compatible images endpoint
      const response = await venice.images.generate({
        model: 'flux-dev',
        prompt: fullPrompt,
        n: 1,
        size: `${width || 1024}x${height || 1024}` as '1024x1024',
      });

      const imageData = response.data?.[0];
      const imageUrl = imageData?.url || imageData?.b64_json;

      if (!imageUrl) {
        return {
          content: [
            {
              type: 'text' as const,
              text: 'No image generated',
            },
          ],
          isError: true,
        };
      }

      return {
        content: [
          {
            type: 'text' as const,
            text: `Image generated successfully!\n\nURL: ${imageUrl}\n\nPrompt: ${fullPrompt}${negative_prompt ? `\nNegative: ${negative_prompt}` : ''}`,
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice image error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ============================================
// EMBEDDINGS
// ============================================

server.tool(
  'venice_embeddings',
  'Generate text embeddings using Venice AI.',
  {
    text: z.string().describe('Text to generate embeddings for'),
    model: z.string().optional().describe('Embedding model. Default: text-embedding-3-small'),
  },
  async (params: { text: string; model?: string }) => {
    try {
      const { text, model } = params;

      const response = await venice.embeddings.create({
        model: model || 'text-embedding-3-small',
        input: text,
      });

      const embedding = response.data[0]?.embedding;

      return {
        content: [
          {
            type: 'text' as const,
            text: `Embedding generated (${embedding?.length || 0} dimensions)\n\nFirst 10 values: [${embedding?.slice(0, 10).join(', ')}...]`,
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice embeddings error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ============================================
// UTILITY TOOLS
// ============================================

server.tool(
  'venice_models',
  'List available Venice AI models and their capabilities.',
  {},
  async () => {
    const models = `
Venice AI Models:

TEXT MODELS:
1. llama-3.3-70b (Default)
   - General conversation
   - 128K context window
   - Best for: Chat, analysis, general tasks

2. deepseek-r1-671b
   - Advanced reasoning model
   - Chain-of-thought capabilities
   - Best for: Math, logic, complex problems

3. qwen-2.5-coder-32b
   - Code specialist
   - Strong at programming tasks
   - Best for: Code generation, review, debugging

4. dolphin-2.9.3-mistral-7b
   - Uncensored model
   - No content restrictions
   - Best for: Creative writing, unrestricted queries

IMAGE MODELS:
1. flux-dev
   - High-quality image generation
   - Photorealistic capabilities
   - Best for: Art, design, visualization

Key Venice Features:
- Privacy-first: No data storage or logging
- Uncensored models available
- Web search integration (enable_web_search)
- OpenAI-compatible API

API: https://api.venice.ai/api/v1
Docs: https://docs.venice.ai
`;

    return {
      content: [
        {
          type: 'text' as const,
          text: models.trim(),
        },
      ],
    };
  }
);

server.tool(
  'venice_json',
  'Get structured JSON output from Venice.',
  {
    prompt: z.string().describe('Describe what JSON structure you need'),
    schema: z.string().optional().describe('Optional JSON schema hint'),
  },
  async (params: { prompt: string; schema?: string }) => {
    try {
      const { prompt, schema } = params;

      let systemPrompt = 'You are a JSON generator. Output ONLY valid JSON, no markdown, no explanation.';
      if (schema) {
        systemPrompt += ` Follow this schema: ${schema}`;
      }

      const response = await venice.chat.completions.create({
        model: 'llama-3.3-70b',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: prompt },
        ],
        temperature: 0,
        response_format: { type: 'json_object' },
      });

      return {
        content: [
          {
            type: 'text' as const,
            text: response.choices[0]?.message?.content || '{}',
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  'venice_summarize',
  'Summarize text using Venice AI.',
  {
    text: z.string().describe('Text to summarize'),
    style: z.enum(['brief', 'detailed', 'bullets']).optional().describe('Summary style. Default: brief'),
  },
  async (params: { text: string; style?: string }) => {
    try {
      const { text, style } = params;

      const stylePrompts: Record<string, string> = {
        brief: 'Summarize in 2-3 sentences.',
        detailed: 'Provide a detailed summary covering all key points.',
        bullets: 'Summarize as bullet points.',
      };

      const response = await venice.chat.completions.create({
        model: 'llama-3.3-70b',
        messages: [
          { role: 'user', content: `${stylePrompts[style || 'brief']}\n\nText:\n${text}` },
        ],
        temperature: 0.3,
      });

      return {
        content: [
          {
            type: 'text' as const,
            text: response.choices[0]?.message?.content || 'No summary',
          },
        ],
      };
    } catch (error: unknown) {
      const err = error as { message: string };
      return {
        content: [
          {
            type: 'text' as const,
            text: `Venice error: ${err.message}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// ============================================
// START SERVER
// ============================================

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Venice AI MCP server running');
}

main().catch(console.error);
