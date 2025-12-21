# Contributing to AutoNavX

## Team Workflow

### Branch Structure
```
main              # Stable, production-ready code
  └── development # Team integration branch
        ├── feature/sensor-fusion    # Member 1
        ├── feature/rl-agent         # Member 2
        ├── feature/motion-planner   # Member 3
        ├── feature/raim-module      # Member 4
        └── feature/testing          # Member 5
```

### Getting Started (For Each Team Member)

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AutoNavX.git
   cd AutoNavX
   ```

2. **Switch to development branch**
   ```bash
   git checkout development
   ```

3. **Create your feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Daily Workflow

1. **Pull latest changes before starting work**
   ```bash
   git checkout development
   git pull origin development
   git checkout your-feature-branch
   git merge development
   ```

2. **Make your changes and commit regularly**
   ```bash
   git add .
   git commit -m "feat: describe your changes"
   ```

3. **Push your branch**
   ```bash
   git push origin your-feature-branch
   ```

4. **Create a Pull Request on GitHub**
   - Go to the repository on GitHub
   - Click "Pull Requests" → "New Pull Request"
   - Set base: `development`, compare: `your-feature-branch`
   - Request review from at least 1 team member

### Commit Message Convention

Use these prefixes:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:
```
feat: add optical flow to radar processor
fix: resolve collision detection bug
docs: update README with training instructions
```

### Code Review Guidelines

Before merging:
- [ ] Code runs without errors
- [ ] Tests pass (if applicable)
- [ ] Code follows project style
- [ ] At least 1 team member approved

### Team Member Responsibilities

| Member | Primary Module | Files |
|--------|---------------|-------|
| Member 1 | Sensor Fusion | `sensor_fusion.py` |
| Member 2 | RL Agent | `rl_agent.py` |
| Member 3 | Motion Planner | `motion_planner.py` |
| Member 4 | RAIM Module | `raim_module.py` |
| Member 5 | Integration & Testing | `autonavx_agent.py`, tests |

### Merge to Main

Only merge `development` → `main` when:
- All features are tested
- Team agrees code is stable
- No critical bugs

### Useful Git Commands

```bash
# Check status
git status

# View branches
git branch -a

# Switch branch
git checkout branch-name

# Discard local changes
git checkout -- filename

# View commit history
git log --oneline

# Stash changes temporarily
git stash
git stash pop
```

### Setting Up Development Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo to test
python run_autonavx.py --mode demo

# Run training
python train_autonavx.py --episodes 100
```

### Questions?

Create an Issue on GitHub or discuss in your team chat!
