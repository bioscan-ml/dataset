Contributing
============

Thanks for considering contributing to the bioscan-dataset package!

Please take a moment to read these guidelines which will help you contribute
efficiently to the package.


Issues
------

If you have found an issue with the [BIOSCAN-5M data](https://zenodo.org/records/11973457),
such as an invalid image or erroneous label, please
[open an issue on the BIOSCAN-5M repository](https://github.com/bioscan-ml/BIOSCAN-5M/issues/new/choose)
to report it.
If you have found an issue with the [BIOSCAN-1M data](https://zenodo.org/records/8030065),
please check whether this was fixed in BIOSCAN-5M by aligning the dataset
records in the two datasets on the `processid` field before reporting the
issue since some known issues with the BIOSCAN-1M data were fixed in
BIOSCAN-5M.
There is currently no intention to update the BIOSCAN-1M data further, but
BIOSCAN-5M will be periodically updated if issues are identified.

If you encounter a bug with the bioscan-dataset package which you would like
to report, please:
1. First ensure you are using the latest release
   ![PyPI - Version](https://img.shields.io/pypi/v/bioscan-dataset)
   and/or check the [changelog](https://github.com/bioscan-ml/dataset/blob/master/CHANGELOG.rst)
   to see if the issue was already resolved.
2. Secondly, check the [open issues](https://github.com/bioscan-ml/dataset/issues)
   to see if the bug has already been reported.
3. Otherwise, open a [new issue](https://github.com/bioscan-ml/dataset/issues/new/choose),
   describing the version of the package you are using and details about your
   environment, the behaviour you expected, and the behaviour you instead
   encountered.


Pull requests
-------------

Pull requests are welcome!

For enhancements, please discuss with the code owner(s) by slack, email, or a
GitHub issue before authoring your enhancement. This is to check your proposal
is in line with the goals of the package and avoid wasting your time
implementing code which will not be integrated.

If you have found a bug and are willing to fix it yourself, have found an error
in the documentation, or can provide an improvement to the documentation, you
are welcome to open a pull request without discussing it with the code owners
beforehand.

1. Fork the repository.
2. Clone your fork to the machine where you will develop the codebase.
3. Install the pre-commit stack ([see below](#pre-commit) for details).
4. Checkout a new feature or bugfix branch.
    - The branch name should be all in lowercase,
      start with a [commit tag](#commit-messages) (in lowercase),
      followed by a few words joined by hyphens that describe high-level
      objective of the PR.
5. Implement and commit your changes.
    - Commits should be atomic: don't change multiple things that aren't related to each other in the same commit.
    - Commit messages should be succinctly descriptive of the change they contain.
    - Commit messages should open with an appropriate commit tag, or in some cases, combination of commit tags.
      ([See below](#commit-messages) for details.)
    - Refactor your commit history to squash any "oops" commits that fix other commits within your PR into the commit they fix.
6. For new features, consider adding an example usage of the feature to README.rst.
    - Note that you don't need to add details of the new feature to CHANGELOG.rst as this is updated at the time of release.
7. [Submit your PR](https://github.com/bioscan-ml/dataset/compare) to the master branch.
8. A maintainer should review your code within a week.
    - If you haven't heard anything after two weeks, feel free to send a reminder or check-in message.


### pre-commit

The repository comes with a [pre-commit](https://pre-commit.com/) stack.
This is a set of git hooks which are executed every time you make a commit.
The hooks catch errors as they occur by checking your python code is valid and
[flake8](https://flake8.pycqa.org/)-compliant, and will automatically
adjust your code's formatting to conform to a standardized code style
([black](https://github.com/psf/black)).

To set up the pre-commit hooks, run the following shell code from within the repo directory:

```bash
# Install the developmental dependencies
pip install -r requirements-dev.txt
# Install the pre-commit hooks
pre-commit install
```

Whenever you try to commit code which is flagged by the pre-commit
hooks, the commit will *not happen*. Some of the pre-commit hooks
(such as [black](https://github.com/psf/black),
[isort](https://github.com/timothycrosley/isort)) will automatically
modify your code to fix the issues. When this happens, you'll have to
stage the changes made by the commit hooks and then try your commit
again. Other pre-commit hooks will not modify your code and will just
tell you about issues which you'll then have to manually fix.

You can also manually run the pre-commit stack on all the files at any time:
```bash
pre-commit run --all-files
```
This is particularly useful if you already committed some code before
installing pre-commit and need to run the linter on it later.

To force a commit to go through without passing the pre-commit hooks use the `--no-verify` flag:
```bash
git commit --no-verify
```

The pre-commit stack which comes with the template is highly
opinionated, and includes the following operations:

- Code is reformatted to use the [black](https://github.com/psf/black)
  style. Any code inside docstrings will be formatted to black using
  [blackendocs](https://github.com/asottile/blacken-docs).
- Imports are automatically sorted using
  [isort](https://github.com/timothycrosley/isort).
- [flake8](https://flake8.pycqa.org/) is run to check for
  linting errors with [pyflakes](https://github.com/PyCQA/pyflakes)
  (e.g. code does not compile or a variable is used before it is defined),
  and for conformity to the python style guide
  [PEP-8](https://www.python.org/dev/peps/pep-0008/).
- Several [hooks from pre-commit](https://github.com/pre-commit/pre-commit-hooks)
  are used to screen for non-language specific git issues, such as incomplete
  git merges, or overly large files being commited to the repo, etc.
- Several [hooks from pre-commit specific to python](https://github.com/pre-commit/pygrep-hooks)
  are used to screen for rST formatting issues, ensure noqa flags always
  specify an error code to ignore, etc.

Once it is set up, the pre-commit stack will run locally on every commit.
The pre-commit stack will also run on github to ensure PRs are conformal.


### Commit messages

Commit messages should be clear and follow a few basic rules. Example:

    ENH: Add <functionality-X> [to <dataset or method name>]

    The first line of the commit message starts with a capitalized acronym
    (options listed below) indicating what type of commit this is. Then a blank
    line, then more text if needed. Lines shouldn't be longer than 72
    characters. If the commit is related to a ticket, indicate that with
    "See #123", "Closes #123" or similar.

Describing the motivation for a change, the nature of a bug for bug fixes or
some details on what an enhancement does are also good to include in a commit
message. Messages should be understandable without looking at the code changes.
Simple changes need only be one line long without extended description.
A commit message like `MNT: Fixed another one` is an example of what not to do;
the reader has to go look for context elsewhere to understand the message.

Standard acronyms (commit tags) to start the commit message with are based on
the [commit tags used by numpy](https://numpy.org/doc/2.2/dev/development_workflow.html#writing-the-commit-message)
as follows:

    API: an (incompatible) API change
    BUG: bug fix
    CI: continuous integration
    DEP: deprecate something, or remove a deprecated object
    DEV: development tool or utility
    DOC: documentation
    ENH: enhancement
    MNT: maintenance commit (refactoring, typos, etc.)
    REL: related to releasing bioscan-dataset
    REV: revert an earlier commit
    STY: style fix (whitespace, PEP8)
    TST: addition or modification of tests
    WIP: work in progress, do not merge
