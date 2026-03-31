"use client";

import type { Message } from "@langchain/langgraph-sdk";

import type { MessageResponseProps } from "@/components/ai-elements/message";
import {
  extractClarificationArgs,
  extractContentFromMessage,
  type ClarificationArgs,
} from "@/core/messages/utils";
import { cn } from "@/lib/utils";

import { MarkdownContent } from "./markdown-content";

/**
 * Renders an ask_clarification tool message.
 *
 * Primary path: extract the question and options from the original AI
 * message's tool call args and render them as structured HTML elements.
 * This bypasses the markdown/Streamdown pipeline entirely, avoiding issues
 * where the streaming word-animation rehype plugin or the markdown block
 * splitter incorrectly renders array content character-by-character.
 *
 * Fallback path: if tool call args cannot be extracted (e.g. missing AI
 * message), render the tool message content as markdown.
 */
export function ClarificationMessage({
  toolMessage,
  messages,
  isLoading,
  rehypePlugins,
}: {
  toolMessage: Message;
  messages: Message[];
  isLoading: boolean;
  rehypePlugins: MessageResponseProps["rehypePlugins"];
}) {
  // Prefer structured rendering from tool call args — always correct.
  const args = extractClarificationArgs(toolMessage, messages);
  if (args) {
    return <ClarificationFromArgs args={args} />;
  }

  // Fallback: render whatever content the tool message has as markdown.
  const content = extractContentFromMessage(toolMessage);
  if (content) {
    return (
      <MarkdownContent
        content={content}
        isLoading={isLoading}
        rehypePlugins={rehypePlugins}
      />
    );
  }

  return null;
}

const TYPE_ICONS: Record<string, string> = {
  missing_info: "\u2753",
  ambiguous_requirement: "\ud83e\udd14",
  approach_choice: "\ud83d\udd00",
  risk_confirmation: "\u26a0\ufe0f",
  suggestion: "\ud83d\udca1",
};

function ClarificationFromArgs({ args }: { args: ClarificationArgs }) {
  const icon = TYPE_ICONS[args.clarification_type ?? ""] ?? "\u2753";

  return (
    <div className="flex flex-col gap-2">
      {args.context && (
        <p className="text-muted-foreground text-sm">{args.context}</p>
      )}
      <p className="text-foreground text-base">
        {icon} {args.question}
      </p>
      {args.options && args.options.length > 0 && (
        <ol className={cn("ml-6 list-decimal space-y-1")}>
          {args.options.map((option, i) => (
            <li key={i} className="text-foreground text-sm">
              {option}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

