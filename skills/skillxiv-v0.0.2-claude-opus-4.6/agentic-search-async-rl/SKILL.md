---
name: agentic-search-async-rl
title: Beyond Ten Turns - Long-Horizon Agentic Search with Asynchronous RL
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.07976
keywords: [agent-search, reinforcement-learning, long-horizon, asynchronous, autonomous-qa-generation]
description: "Enables long-horizon agentic search extending beyond 100 tool calls through scalable asynchronous RL training with autonomous QA dataset synthesis."
---

## Beyond Ten Turns: Long-Horizon Agentic Search with Asynchronous RL

### Core Concept

ASearcher overcomes the limitation that traditional online RL restricts agents to roughly 10 interaction turns. By employing scalable asynchronous RL training, the system enables agents to conduct searches extending beyond 100 tool calls while maintaining training efficiency. The approach includes autonomous LLM-based synthesis of large-scale QA datasets without external dependencies.

### Architecture Overview

- **Asynchronous RL Framework**: Fully asynchronous training enabling long-horizon search
- **Extended Tool Call Sequence**: Support for 100+ tool interactions per episode
- **Autonomous QA Synthesis**: LLM agents autonomously generate high-quality QA pairs
- **Large-Scale Dataset Generation**: Create comprehensive evaluation benchmarks
- **Zero-Shot Transfer**: Base models work at inference without external LLMs

### Implementation Steps

**Step 1: Design Asynchronous RL Infrastructure**

Create efficient asynchronous training system:

```python
# Pseudocode for asynchronous RL
class AsyncSearchEnvironment(gym.Env):
    def __init__(self, tools, max_steps=200):
        super().__init__()
        self.tools = tools
        self.max_steps = max_steps
        self.current_step = 0
        self.search_history = []

    def step(self, action):
        """
        Execute action and return observation.

        Args:
            action: Tool to call and parameters

        Returns:
            observation: Result from tool
            reward: Reward signal
            done: Whether episode finished
            info: Metadata
        """
        self.current_step += 1

        # Execute tool asynchronously
        tool_name = action['tool']
        tool_params = action['params']

        try:
            result = self._execute_tool_async(tool_name, tool_params)
        except Exception as e:
            result = {'error': str(e)}

        # Track in history
        self.search_history.append({
            'step': self.current_step,
            'action': action,
            'result': result
        })

        # Compute reward
        reward = self._compute_step_reward(result, action)

        # Check termination
        done = (self.current_step >= self.max_steps) or self._should_terminate()

        return result, reward, done, {'step': self.current_step}

    def _execute_tool_async(self, tool_name, params):
        """
        Execute tool asynchronously to enable long horizons.
        """
        import asyncio

        async def async_tool_call():
            tool = self.tools[tool_name]
            return await tool.execute_async(**params)

        # Use event loop for async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_tool_call())

        return result

    def _compute_step_reward(self, result, action):
        """
        Compute reward for this step.
        """
        if 'error' in result:
            return -0.1

        # Reward for useful information
        info_gain = len(str(result)) / 1000.0
        reward = min(info_gain, 0.5)

        return reward

    def _should_terminate(self):
        """
        Decide if search should terminate early.
        """
        # Check if solution found
        if len(self.search_history) >= 3:
            last_results = self.search_history[-3:]
            if all('answer' in r['result'] for r in last_results):
                return True

        return False

    def reset(self):
        """
        Reset environment for new episode.
        """
        self.current_step = 0
        self.search_history = []
        return {}
```

**Step 2: Implement Asynchronous Training Loop**

Create distributed training infrastructure:

```python
# Pseudocode for async RL training
class AsyncRLTrainer:
    def __init__(self, model, environment, num_workers=16):
        super().__init__()
        self.model = model
        self.environment = environment
        self.num_workers = num_workers
        self.replay_buffer = deque(maxlen=100000)

    def collect_experience_async(self):
        """
        Asynchronously collect experience from multiple workers.
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []

            for worker_id in range(self.num_workers):
                future = executor.submit(self._run_worker_episode, worker_id)
                futures.append(future)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                trajectory = future.result()
                self.replay_buffer.append(trajectory)

        return len(self.replay_buffer)

    def _run_worker_episode(self, worker_id):
        """
        Run single episode in worker.
        """
        env = copy.deepcopy(self.environment)
        obs = env.reset()

        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'observations': []
        }

        done = False
        step_count = 0

        while not done and step_count < 200:
            # Agent selects action
            with torch.no_grad():
                action_logits = self.model(obs)
                action = torch.multinomial(F.softmax(action_logits, dim=-1), 1)

            # Environment step
            obs, reward, done, info = env.step(self._action_to_dict(action))

            # Store trajectory
            trajectory['states'].append(obs)
            trajectory['actions'].append(action.item())
            trajectory['rewards'].append(reward)
            trajectory['observations'].append(obs)

            step_count += 1

        # Compute returns
        trajectory['returns'] = self._compute_returns(trajectory['rewards'])

        return trajectory

    def _compute_returns(self, rewards):
        """
        Compute discounted returns.
        """
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        return returns

    def train_on_batch(self, batch_size=128):
        """
        Train on batch from replay buffer.
        """
        if len(self.replay_buffer) < batch_size:
            return None

        batch = random.sample(self.replay_buffer, batch_size)

        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        for trajectory in batch:
            states = torch.tensor(trajectory['states'])
            actions = torch.tensor(trajectory['actions'])
            returns = torch.tensor(trajectory['returns'])

            # Forward pass
            logits = self.model(states)
            log_probs = F.log_softmax(logits, dim=-1)

            # Policy gradient loss
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1))
            loss = -(action_log_probs * returns.unsqueeze(-1)).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        return loss.item()

    def train(self, num_iterations=1000):
        """
        Full async RL training loop.
        """
        for iteration in range(num_iterations):
            # Collect experience asynchronously
            num_trajectories = self.collect_experience_async()

            # Train on batch
            for _ in range(10):  # Multiple training iterations per collection
                loss = self.train_on_batch()

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}: Buffer size = {num_trajectories}")

        return self.model
```

**Step 3: Implement Autonomous QA Generation**

Generate training questions autonomously:

```python
# Pseudocode for autonomous QA synthesis
class AutonomousQAGenerator:
    def __init__(self, qa_generation_model, synthesizer_tools):
        super().__init__()
        self.qa_model = qa_generation_model
        self.tools = synthesizer_tools

    def generate_question(self, seed_topic):
        """
        Generate a new question based on seed topic.

        Args:
            seed_topic: Domain or topic for question

        Returns:
            question: Generated question
        """
        prompt = f"""Generate a challenging question about {seed_topic} that requires web search to answer.
        The question should be specific enough to require at least 3-5 search steps to answer."""

        with torch.no_grad():
            question = self.qa_model.generate(
                prompt,
                max_length=100,
                temperature=0.8
            )

        return question

    def answer_question_autonomously(self, question):
        """
        Autonomously search and answer a question.

        Args:
            question: Question to answer

        Returns:
            answer: Generated answer
            search_path: Search actions taken
        """
        search_path = []
        search_queries = self._decompose_question(question)

        collected_info = []

        for query in search_queries[:5]:  # Limit to 5 search steps
            # Search
            results = self.tools['search'](query)
            search_path.append({'action': 'search', 'query': query})

            # Read relevant pages
            for result in results[:2]:
                try:
                    page_content = self.tools['read_page'](result['url'])
                    collected_info.append(page_content)
                    search_path.append({'action': 'read', 'url': result['url']})
                except:
                    pass

        # Synthesize answer
        answer = self._synthesize_answer(question, collected_info)

        return answer, search_path

    def _decompose_question(self, question):
        """
        Break question into search queries.
        """
        prompt = f"""Break down this question into 3-5 search queries:
        {question}

        Return as comma-separated list."""

        with torch.no_grad():
            queries_text = self.qa_model.generate(prompt, max_length=200)

        queries = [q.strip() for q in queries_text.split(',')]
        return queries

    def _synthesize_answer(self, question, information):
        """
        Generate answer from collected information.
        """
        context = '\n'.join(information)

        prompt = f"""Question: {question}

Information collected:
{context}

Please provide a clear, concise answer based on the information above."""

        with torch.no_grad():
            answer = self.qa_model.generate(prompt, max_length=200)

        return answer

    def generate_qa_dataset(self, topics, num_per_topic=10):
        """
        Generate large-scale QA dataset autonomously.

        Args:
            topics: List of seed topics
            num_per_topic: Questions to generate per topic

        Returns:
            dataset: QA dataset
        """
        dataset = []

        for topic in topics:
            for _ in range(num_per_topic):
                try:
                    # Generate question
                    question = self.generate_question(topic)

                    # Answer autonomously
                    answer, search_path = self.answer_question_autonomously(question)

                    dataset.append({
                        'question': question,
                        'answer': answer,
                        'search_path': search_path,
                        'topic': topic
                    })

                except Exception as e:
                    print(f"Error generating QA for {topic}: {e}")

        return dataset
```

**Step 4: Implement Long-Horizon Search Evaluation**

Evaluate agent performance on extended searches:

```python
# Pseudocode for long-horizon evaluation
class LongHorizonSearchEvaluator:
    def __init__(self, agent, tools):
        super().__init__()
        self.agent = agent
        self.tools = tools

    def evaluate_agent(self, test_questions, max_steps=150):
        """
        Evaluate agent on test questions with long horizons.

        Args:
            test_questions: List of questions to answer
            max_steps: Maximum steps allowed

        Returns:
            results: Evaluation metrics
        """
        results = {
            'success_rate': 0,
            'avg_steps': 0,
            'answer_quality': 0,
            'long_horizon_success': 0
        }

        correct_answers = 0
        total_steps = 0
        long_horizon_attempts = 0

        for question in test_questions:
            # Initialize search
            obs = {'question': question, 'search_history': []}

            answer = None
            step_count = 0
            found_answer = False

            # Run search
            while step_count < max_steps:
                # Agent decides next action
                with torch.no_grad():
                    action = self.agent.decide_action(obs)

                if action['type'] == 'synthesize':
                    answer = action.get('answer')
                    found_answer = True
                    break

                # Execute action
                if action['type'] == 'search':
                    results_list = self.tools['search'](action['query'])
                    obs['search_history'].append({
                        'type': 'search',
                        'query': action['query'],
                        'results': results_list
                    })

                elif action['type'] == 'read_page':
                    content = self.tools['read_page'](action['url'])
                    obs['search_history'].append({
                        'type': 'read',
                        'url': action['url'],
                        'content': content
                    })

                step_count += 1

            if found_answer:
                correct_answers += 1

            total_steps += step_count

            # Track long-horizon success (>30 steps)
            if step_count > 30:
                long_horizon_attempts += 1

        # Compute metrics
        results['success_rate'] = correct_answers / len(test_questions)
        results['avg_steps'] = total_steps / len(test_questions)
        results['long_horizon_success'] = long_horizon_attempts

        return results
```

### Practical Guidance

**Hyperparameters and Configuration**:
- Number of async workers: 8-32 depending on infrastructure
- Maximum steps per episode: 100-200
- Replay buffer size: 50k-200k trajectories
- RL learning rate: 1e-5 to 5e-5
- Training iterations: 500-2000

**When to Use Long-Horizon Agentic Search**:
- Complex multi-step research and information retrieval
- Scenarios requiring iterative refinement and exploration
- Systems where agent must autonomously discover solution paths
- Applications with 30+ step search horizons

**When NOT to Use**:
- Simple factual lookup (can be answered in <5 steps)
- Scenarios with unreliable or sparse information sources
- Real-time systems with strict step budgets
- When deterministic paths are more reliable than learned policies

**Implementation Notes**:
- Asynchronous training critical for handling long episodes
- QA dataset generation should maintain diversity across domains
- Monitor success rates at different horizon lengths (10, 50, 100+ steps)
- Consider curriculum learning: start with short horizons, gradually increase
- Autonomous QA generation requires periodic manual validation

### Reference

Paper: Beyond Ten Turns: Long-Horizon Agentic Search with Asynchronous RL
ArXiv: 2508.07976
Performance: Enables 100+ tool calls and 400,000+ output tokens during training, enables agents to develop sophisticated multi-step search strategies
