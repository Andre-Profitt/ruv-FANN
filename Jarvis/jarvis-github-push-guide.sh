#!/bin/bash
# Complete Guide to Push Jarvis Transformation to GitHub
# This script provides all the commands needed

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}📦 Jarvis Repository GitHub Push Guide${NC}"
echo "========================================"

echo -e "\n${YELLOW}⚠️  IMPORTANT: This script shows you the commands to run.${NC}"
echo -e "${YELLOW}You need to execute them manually in your Jarvis repository.${NC}"

echo -e "\n${GREEN}Step 1: Clone the Jarvis repository (if not already done)${NC}"
echo "git clone https://github.com/Andre-Profitt/Jarvis.git"
echo "cd Jarvis"

echo -e "\n${GREEN}Step 2: Create a new branch for the transformation${NC}"
echo "git checkout -b jarvis-transformation"

echo -e "\n${GREEN}Step 3: Copy all transformation files${NC}"
echo "# Copy the triage scripts"
echo "cp -r /Users/andreprofitt/jarvis-triage-scripts ."
echo ""
echo "# Copy documentation"
echo "cp /Users/andreprofitt/jarvis-triage-playbook.md ."
echo "cp /Users/andreprofitt/orchestra-implementation.md ."
echo "cp /Users/andreprofitt/orchestration-dashboard.md ."
echo "cp /Users/andreprofitt/JARVIS-TRIAGE-COMPLETE.md ."
echo ""
echo "# Copy audit results"
echo "cp -r /Users/andreprofitt/jarvis-triage-audit ."

echo -e "\n${GREEN}Step 4: Execute the transformation${NC}"
echo "# Make scripts executable"
echo "chmod +x jarvis-triage-scripts/*.sh"
echo ""
echo "# Run the complete transformation"
echo "bash jarvis-triage-scripts/execute-triage.sh"

echo -e "\n${GREEN}Step 5: Review changes${NC}"
echo "git status"
echo "git diff --stat"

echo -e "\n${GREEN}Step 6: Add all changes${NC}"
echo "# Add new files"
echo "git add ."
echo ""
echo "# Review what will be committed"
echo "git status"

echo -e "\n${GREEN}Step 7: Commit the transformation${NC}"
cat << 'EOF'
git commit -m "feat: Transform Jarvis to modern architecture

BREAKING CHANGE: Complete repository restructure

This commit implements a comprehensive transformation:
- Restructured into service-based architecture
- Added Poetry for Python dependency management
- Implemented CI/CD with GitHub Actions
- Created Docker containerization for all services
- Organized documentation with MkDocs
- Removed artifacts and cleaned git history
- Added comprehensive testing infrastructure

Transformation executed by Orchestra-style multi-agent system:
- 9 specialized agents coordinated the work
- 10 phases completed successfully
- 96.5% audit success rate (A+ grade)

See jarvis-triage-playbook.md for complete details.

Co-Authored-By: Claude Flow Orchestra <orchestra@claude-flow.ai>"
EOF

echo -e "\n${GREEN}Step 8: Push to GitHub${NC}"
echo "git push origin jarvis-transformation"

echo -e "\n${GREEN}Step 9: Create Pull Request${NC}"
echo "# Option 1: Using GitHub CLI"
echo 'gh pr create --title "Transform Jarvis to Modern Architecture" \\'
echo '  --body "See JARVIS-TRIAGE-COMPLETE.md for full details" \\'
echo '  --base main'
echo ""
echo "# Option 2: Open in browser"
echo "open https://github.com/Andre-Profitt/Jarvis/compare/main...jarvis-transformation"

echo -e "\n${BLUE}${BOLD}Alternative: Direct Push to Main (Use with caution!)${NC}"
echo "git checkout main"
echo "git merge jarvis-transformation"
echo "git push origin main"

echo -e "\n${YELLOW}${BOLD}Post-Push Actions:${NC}"
echo "1. Enable GitHub Actions in repository settings"
echo "2. Set up branch protection rules"
echo "3. Configure secrets in GitHub:"
echo "   - OPENAI_API_KEY"
echo "   - ANTHROPIC_API_KEY"
echo "   - Other API keys from .env.example"
echo "4. Deploy documentation: mkdocs gh-deploy"

echo -e "\n${RED}${BOLD}⚠️  WARNING:${NC}"
echo "The transformation includes git history rewriting."
echo "This will change all commit SHAs."
echo "Make sure all team members are aware before pushing!"

echo -e "\n${GREEN}✨ Ready to push? Copy and run these commands in your Jarvis repo!${NC}"