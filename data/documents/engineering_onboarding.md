# Acme Corporation - Onboarding Guide for New Engineers

## First Week Checklist

Welcome to the engineering team at Acme Corporation! During your first week, please complete the following steps:

1. Set up your development environment using the Engineering Setup Guide located on the internal wiki at wiki.acmecorp.com/engineering/setup.
2. Request access to the company GitHub organization by emailing devops@acmecorp.com with your GitHub username.
3. Complete the mandatory Security Awareness Training module in the Learning Management System (LMS). This must be finished within your first 5 business days.
4. Schedule a 1-on-1 with your assigned mentor. Your mentor will be introduced to you by your manager on your first day.
5. Join the following Slack channels: #engineering, #engineering-help, #deploy-notifications, and your team-specific channel.

## Development Standards

All code at Acme Corporation must follow these standards:

- Code must be reviewed by at least one peer before merging to the main branch.
- All pull requests must include unit tests with a minimum of 80% code coverage for new code.
- Commit messages must follow the Conventional Commits format (e.g., "feat: add user login endpoint").
- All services must include health check endpoints at /health and /ready.
- API documentation must be maintained in OpenAPI 3.0 format.

## Deployment Process

Acme uses a continuous delivery pipeline managed through Jenkins. The deployment process works as follows:

1. Developers push code to a feature branch and open a pull request.
2. Automated tests run in the CI pipeline. All tests must pass before merging.
3. After merge to main, the code is automatically deployed to the staging environment.
4. QA engineers perform manual testing on staging for 24 hours.
5. After QA approval, the release manager promotes the build to production using the deployment dashboard at deploy.acmecorp.com.
6. Production deployments are only allowed Monday through Thursday, between 10:00 AM and 2:00 PM EST. No deployments are permitted on Fridays or the day before company holidays.

## On-Call Rotation

All engineers participate in the on-call rotation. Each rotation lasts one week, from Monday 9:00 AM to the following Monday 9:00 AM. On-call engineers must:

- Acknowledge production alerts within 15 minutes
- Begin investigation within 30 minutes
- Escalate to the team lead if the issue is not resolved within 2 hours
- Write a post-mortem document for any incident lasting longer than 1 hour

On-call compensation is an additional $500 per week. If called outside business hours, engineers receive an additional $100 per incident responded to.
