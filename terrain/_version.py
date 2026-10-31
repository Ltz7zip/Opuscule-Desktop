"""版本号的唯一来源。

只允许以下格式（``X`` 为非负整数，``N`` 从 1 开始递增）：

    X.X.X-dev.N     开发版
    X.X.X-alpha.N   内测版
    X.X.X-beta.N    公测版
    X.X.X-rc.N      候选版
    X.X.X           正式版（GA）

阶段顺序固定为 dev → alpha → beta → rc → GA；不得跳级，也不得混用。
其它模块一律 ``from terrain._version import __version__``，不要另写版本字符串。
"""

__version__ = "12.0.0"
