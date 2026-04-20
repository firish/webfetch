# Project Context

## What This Is
A lightweight, open-source Python library that replicates and replaces the 
web search tool calls that LLM APIs (Claude, OpenAI, etc.) make internally -
at a fraction of the cost by owning the full pipeline.

Instead of paying $15–35/1k queries for hosted LLM web search tools, this 
library lets you:
1. Fetch search results yourself via a pluggable search adapter
2. Extract clean text from result pages
3. Rank and select the most relevant chunks using a multi-stage reranker
4. Send only trimmed, relevant context to the LLM

The LLM then only pays for reasoning/extraction tokens - not search overhead.

## Problem Statement
LLM-hosted web search tools (Claude web search, OpenAI browsing, Perplexity) 
bundle crawling, indexing, fetching, and ranking into a single opaque API 
call. This is convenient but expensive, unobservable, and not customizable.

This library deconstructs that pipeline into explicit, swappable stages so 
developers can: own costs, inspect intermediate results, swap components, 
cache aggressively, and learn how production search systems work.

## Primary Use Case (Reference Implementation)
**Calibration and Test & Measurement equipment spec lookup**

- Input: manufacturer name + model number + (optionally asset type)
- Equipment: scales, torque tools, TRGs, TPGs, calipers, pin gages, 
  ohmmeters, and similar precision instruments
- Output: structured JSON specs - range, accuracy, resolution, uncertainty, 
  calibration interval, units, IP rating, etc.
- Volume: hundreds of lookups/day, thousands/month
- Key property: specs are stable - heavy caching opportunity

This use case drives the design decisions but the library is fully 
general-purpose. Any domain that currently uses LLM web search tools is 
a valid target.

## Cost Model
| Approach | Cost at ~3k queries/month |
|---|---|
| Hosted LLM web search tool | $50–200+/month |
| This library (Brave + Haiku) | ~$10–30/month |
| This library (DDG + Haiku, cached) | ~$3–10/month |

Savings compound with caching - identical queries (same mfr+model) are 
free after the first fetch.

## Scope
**In scope:**
- HTML page fetching and clean text extraction
- PDF datasheet extraction
- Multi-stage semantic reranking of retrieved chunks
- Token budget management before LLM call
- Pluggable search provider adapters
- Result caching layer
- Structured JSON output via a cheap LLM extraction call

**Out of scope (for now):**
- Building or maintaining a search index
- JavaScript-heavy SPA rendering (Playwright is an optional fallback only)
- Authentication / paywalled content
- Real-time / streaming results

## Target Users
- Developers currently paying for LLM web search tool calls at scale
- Researchers building RAG or information extraction pipelines
- Anyone who wants to understand how production LLM search works internally

## Status
> Last updated: project inception
> Update this section whenever goals, scope, or use cases change.