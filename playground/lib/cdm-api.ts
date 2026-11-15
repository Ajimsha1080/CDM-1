/**
 * CDM-1 API Client
 * Typed REST client connecting Playground to the local CDM-1 engine.
 */

export interface DecisionRequest {
  text: string;
  structured_features?: Record<string, any>;
  task?: "binary" | "choice" | "score" | "predict" | "priority" | "risk";
  candidates?: string[];
  prohibited_actions?: string[];
}

export interface PredictedOutcome {
  binary_outcome_probability: number;
  predicted_resolution_time: number;
}

export interface DecisionResponse {
  decision_id: string;
  model: string;
  task: string;
  decision: any;
  confidence: number;
  calibrated: boolean;
  score?: number | null;
  neural_score?: number | null;
  heuristic_score?: number | null;
  used_neural_inference: boolean;
  priority: string;
  risk: number;
  abstain: boolean;
  selected_candidate?: string | null;
  candidate_scores?: Record<string, number | null> | null;
  predicted_outcome: PredictedOutcome;
  tenant_id: string;
  latency_ms: number;
}

export interface CognitiveProcessRequest {
  text: string;
  structured_features?: Record<string, any>;
  task?: string;
  candidates?: string[];
  prohibited_actions?: string[];
  goals?: string[];
  constraints?: string[];
  session_id?: string;
}

export interface CognitiveProcessResponse {
  execution_id: string;
  decision: any;
  request_id?: string;
  execution_status?: "authorized" | "proposed" | "needs_approval" | "abstained" | "rejected" | string;
  proposed_decision?: any;
  execution_authorized?: boolean;
  abstention_status?: "none" | "abstained" | "policy_rejection" | "uncertainty" | "missing_evidence" | string;
  abstention_reason?: string | null;
  confidence: number;
  calibrated_confidence: number;
  uncertainty?: number;
  assessed_risk?: number;
  priority: string;
  risk: number;
  operational_score?: number | null;
  candidate_scores?: Record<string, number> | null;
  candidate_rankings?: Array<{ candidate: string; score: number | null; rank: number; is_prohibited: boolean }>;
  policy_check?: {
    passed: boolean;
    violated_rules: string[];
    checked_rules: string[];
    policy_status: string;
  };
  approval_requirements?: {
    required: boolean;
    approver_role?: string | null;
    approval_status: string;
    reason?: string | null;
    configured_supervisor?: string | null;
  };
  verification?: {
    verified: boolean;
    verification_status: string;
    checks: Array<{ check: string; passed: boolean }>;
  };
  supporting_evidence?: Record<string, any>;
  cognitive_trace?: Array<{
    stage_name: string;
    execution_status: string;
    duration_ms: number;
    inputs: Record<string, any>;
    outputs: Record<string, any>;
    errors: string[];
    [key: string]: any;
  }>;
  errors?: string[];
  is_calibrated?: boolean;
  confidence_origin?: string;
  risk_origin?: string;

  // Cognitive Answer Synthesis Fields
  answer?: string;
  answer_type?: string;
  explanation?: string;
  safety_status?: string;
  limitations?: string[];
  important_assumptions?: string[];
  recommended_next_steps?: string[];
  retrieved_external_facts?: Array<Record<string, any>>;
  alternatives?: Array<Record<string, any>> | null;
  used_neural_inference?: boolean;
  predicted_outcomes: Record<string, any>;
  selected_plan?: {
    plan_id: string;
    goal: string;
    steps: Array<{
      step_id: string;
      action: string;
      parameters?: Record<string, any>;
      estimated_risk?: number;
      requires_approval?: boolean;
    }>;
    approved?: boolean;
    prohibited?: boolean;
    abstain?: boolean;
    is_executable?: boolean;
    requires_human_approval?: boolean;
    required_approver_role?: string | null;
  } | null;
  plan?: Record<string, any> | null;
  reasoning_evidence: {
    rule_activations?: Array<{ rule_id: string; weight: number; matched_text?: string }>;
    perceived_entities?: string[];
    risk_factors?: string[];
    safety_violations?: string[];
    [key: string]: any;
  };
  world_state_summary: Record<string, any>;
  abstain: boolean;
  execution_recommendation: string;
  embedding_mode?: string;
  latency_ms: number;
}

export interface AgentRunStep {
  step_index: number;
  state: string;
  plan_step_id?: string;
  action_name: string;
  parameters: Record<string, any>;
  tool_result?: any;
  tool_success?: boolean;
  observation?: string;
  risk_assessed: number;
  requires_approval: boolean;
  approval_status: string;
  operator_reason?: string;
  error?: string;
  timestamp: number;
}

export interface AgentRunResponse {
  run_id: string;
  tenant_id: string;
  session_id: string;
  goal: string;
  task_description: string;
  status: "idle" | "running" | "paused_approval_required" | "completed" | "failed" | "terminated_budget_exceeded" | "abstained";
  step_count: number;
  step_budget: number;
  cost_budget_usd: number;
  accumulated_cost_usd: number;
  accumulated_tokens: number;
  tool_call_count: number;
  history: AgentRunStep[];
  pending_action?: {
    action: string;
    parameters: Record<string, any>;
    step_id?: string;
    estimated_risk: number;
  } | null;
  pending_reason?: string | null;
  created_at: number;
  updated_at: number;
}

export interface CreateAgentRunRequest {
  goal: string;
  task_description?: string;
  step_budget?: number;
  cost_budget_usd?: number;
  tenant_policies?: string[];
  prohibited_actions?: string[];
  verification_spec?: Record<string, any>;
  auto_run?: boolean;
}

export interface MemoryRecord {
  memory_id: string;
  tenant_id: string;
  session_id?: string;
  task?: string;
  content: string;
  embedding?: number[];
  outcome_quality?: number;
  created_at: number;
}

export interface MemoryRecordsResponse {
  tenant_id: string;
  episodic: MemoryRecord[];
  semantic_rules: Array<{
    rule_id: string;
    condition: string;
    action: string;
    weight: number;
    description: string;
  }>;
  procedural: Array<{
    procedure_id: string;
    name: string;
    steps: string[];
    success_rate: number;
  }>;
}

export interface WorldStateResponse {
  tenant_id: string;
  session_id: string;
  world_state: {
    active_goals?: string[];
    active_constraints?: string[];
    observed_facts?: Record<string, any>;
    entity_states?: Record<string, any>;
    metrics?: Record<string, any>;
  };
}

export interface DiagnosticsResponse {
  runtime: {
    python_version: string;
    platform: string;
    torch_version: string;
    cuda_available: boolean;
    device: string;
  };
  model: {
    initialized: boolean;
    mode: "neural" | "rule_based_fallback";
    is_fallback: boolean;
    parameters: string;
    model_path: string;
    weights_exist: boolean;
    manifest_exist: boolean;
    weights_sha256?: string | null;
    load_error?: string | null;
  };
  databases: {
    api_keys: boolean;
    metering: boolean;
    memory: boolean;
    agent_runs: boolean;
  };
  environment: {
    is_production: boolean;
    data_dir: string;
  };
}

export interface ReadinessResponse {
  status: string;
  model_loaded: boolean;
  mode: string;
  weights_verified: boolean;
  manifest_signature_verified: boolean;
  weights_sha256?: string | null;
  parameters: string;
  timestamp: number;
}

// Client Class
class CDMApiClient {
  private baseUrl: string;
  private apiKey: string | null = null;

  constructor() {
    this.baseUrl = typeof window !== "undefined" && window.location.origin.includes("3000")
      ? "http://127.0.0.1:8000"
      : (process.env.NEXT_PUBLIC_CDM_API_URL || "http://127.0.0.1:8000");
  }

  public setBaseUrl(url: string) {
    this.baseUrl = url.replace(/\/$/, "");
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  public setApiKey(key: string) {
    this.apiKey = key.trim();
    if (typeof window !== "undefined") {
      localStorage.setItem("cdm_api_key", this.apiKey);
    }
  }

  public getApiKey(): string | null {
    if (!this.apiKey && typeof window !== "undefined") {
      this.apiKey = localStorage.getItem("cdm_api_key");
    }
    return this.apiKey;
  }

  public async getOrFetchDevKey(): Promise<string> {
    const existing = this.getApiKey();
    if (existing) return existing;

    try {
      const res = await fetch(`${this.baseUrl}/v1/dev/default-key`);
      if (res.ok) {
        const data = await res.json();
        if (data.api_key) {
          this.setApiKey(data.api_key);
          return data.api_key;
        }
      }
    } catch {
      // Fallback
    }

    const defaultDev = "cdm_dev_playground_key";
    this.setApiKey(defaultDev);
    return defaultDev;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const apiKey = await this.getOrFetchDevKey();
    const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : "/" + endpoint}`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errDetail = `HTTP ${response.status} ${response.statusText}`;
        try {
          const errBody = await response.json();
          if (errBody.detail) {
            errDetail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
          } else if (errBody.error) {
            errDetail = errBody.error;
          }
        } catch {
          // Keep raw status text
        }
        throw new Error(errDetail);
      }

      return (await response.json()) as T;
    } catch (err: any) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        throw new Error(`Request to ${endpoint} timed out after 30 seconds.`);
      }
      throw err;
    }
  }

  // System
  public async getHealth(): Promise<{ status: string; timestamp: number }> {
    const res = await fetch(`${this.baseUrl}/v1/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  }

  public async getReadiness(): Promise<ReadinessResponse> {
    const res = await fetch(`${this.baseUrl}/v1/ready`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Readiness probe failed (${res.status})`);
    }
    return res.json();
  }

  public async getVersion(): Promise<Record<string, any>> {
    const res = await fetch(`${this.baseUrl}/v1/version`);
    if (!res.ok) throw new Error("Failed to fetch version info");
    return res.json();
  }

  public async getDiagnostics(): Promise<DiagnosticsResponse> {
    const res = await fetch(`${this.baseUrl}/v1/diagnostics`);
    if (!res.ok) throw new Error("Failed to fetch system diagnostics");
    return res.json();
  }

  // Decisions
  public async createDecision(req: DecisionRequest): Promise<DecisionResponse> {
    return this.request<DecisionResponse>("/v1/decisions", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  public async classify(req: DecisionRequest): Promise<DecisionResponse> {
    return this.request<DecisionResponse>("/v1/classify", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  public async score(req: DecisionRequest): Promise<DecisionResponse> {
    return this.request<DecisionResponse>("/v1/score", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  public async choice(req: DecisionRequest): Promise<DecisionResponse> {
    return this.request<DecisionResponse>("/v1/choice", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  public async predict(req: DecisionRequest): Promise<DecisionResponse> {
    return this.request<DecisionResponse>("/v1/predict", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  // Cognitive Engine
  public async processCognitive(req: CognitiveProcessRequest): Promise<CognitiveProcessResponse> {
    return this.request<CognitiveProcessResponse>("/v1/cognitive/process", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  // Agent Runs
  public async listAgentRuns(limit = 50): Promise<AgentRunResponse[]> {
    return this.request<AgentRunResponse[]>(`/v1/agent/runs?limit=${limit}`);
  }

  public async createAgentRun(req: CreateAgentRunRequest): Promise<AgentRunResponse> {
    return this.request<AgentRunResponse>("/v1/agent/runs", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  public async getAgentRun(runId: string): Promise<AgentRunResponse> {
    return this.request<AgentRunResponse>(`/v1/agent/runs/${encodeURIComponent(runId)}`);
  }

  public async approveAgentRun(runId: string): Promise<AgentRunResponse> {
    return this.request<AgentRunResponse>(`/v1/agent/runs/${encodeURIComponent(runId)}/approve`, {
      method: "POST",
    });
  }

  public async rejectAgentRun(runId: string, reason = "Rejected by human operator"): Promise<AgentRunResponse> {
    return this.request<AgentRunResponse>(`/v1/agent/runs/${encodeURIComponent(runId)}/reject`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    });
  }

  // Memory & World State
  public async getMemoryRecords(limit = 50): Promise<MemoryRecordsResponse> {
    return this.request<MemoryRecordsResponse>(`/v1/memory/records?limit=${limit}`);
  }

  public async getWorldState(sessionId = "default_session"): Promise<WorldStateResponse> {
    return this.request<WorldStateResponse>(`/v1/world-state?session_id=${encodeURIComponent(sessionId)}`);
  }
}

export const cdmApi = new CDMApiClient();
