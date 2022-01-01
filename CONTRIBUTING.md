# How to contribute

We're really glad you're reading this, because we need volunteer developers to help this project come to fruition.
Contributions are highly welcome. 
We recommend to create an issue and discuss your idea before starting lengthy contributions.


## Submitting changes

Please send a [Pull Request](https://github.com/basf/mopti/pull/new/master) with a clear list of what you've done (read more about [pull requests](http://help.github.com/pull-requests/)). 
We can always use more test coverage. 
Please follow our coding conventions (below) and make sure all of your commits are atomic (one feature per commit).


## Development

We're using pre-commit with black, flake8 and isort.
You can install the pre-commit hook with
```
pre-commit install
```


## Publishing

We have a github action set up to publish on PyPI.
The action is triggered on tagged commits, where the tag starts with a "v".

Example:
```
git commit -m "Release v0.10.1: stuff"
git tag -a "v0.10.1" -m "Release v0.10.1"
git push --follow-tags
```


## Documentation

We use mkdocs for documentation.
The docs are built by a github action, which is triggered on commiting to the "docs" branch. 
