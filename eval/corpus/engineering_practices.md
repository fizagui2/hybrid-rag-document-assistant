# Code Review Process

All code changes must be reviewed and approved by at least one other engineer before merging to the main branch. Pull requests should include a description of the change and, where applicable, a link to the related ticket.

# Deployment Process

Deployments to production occur every weekday at 2 PM. Emergency hotfixes may be deployed outside this window with approval from an engineering lead. All deployments must pass the automated test suite before release.

# On-Call Rotation

Engineers participate in an on-call rotation lasting one week at a time. On-call engineers are responsible for responding to production incidents within 15 minutes of an alert. The on-call schedule is published one month in advance.

# Testing Standards

New features require unit test coverage of at least 80% before merging. Integration tests are required for any change that touches a public API endpoint.

# Incident Response

When a production incident occurs, the on-call engineer opens an incident channel and notifies the engineering lead. A post-incident review is required within 3 business days of resolution.
