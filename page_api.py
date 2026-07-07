"""
官方插件 Page API 适配层（参考 astrbot_plugin_livingmemory/core/page_api.py 的写法）。

只暴露一个资源：overrides.json 的读 / 写。UI 设置页 (pages/settings) 通过
AstrBot 官方 Pages 机制的 postMessage 桥（window.AstrBotPluginPage）调用这两个
接口，不需要插件自己处理鉴权——鉴权由 AstrBot Dashboard 在父页面里用当前登录
会话代为发起请求。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quart import request

if TYPE_CHECKING:
    from .main import TriggerGuard

PLUGIN_NAME = "astrbot_plugin_trigger_guard"
PAGE_API_PREFIX = f"/{PLUGIN_NAME}/page"


def _ok(data: Any = None) -> dict[str, Any]:
    return {"status": "ok", "data": data}


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "message": str(message)}


class TriggerGuardPageApi:
    """TriggerGuard 官方插件页面 API。"""

    def __init__(self, plugin: "TriggerGuard") -> None:
        self.plugin = plugin

    def register_routes(self) -> None:
        register = self.plugin.context.register_web_api
        register(
            f"{PAGE_API_PREFIX}/overrides",
            self.get_overrides,
            ["GET"],
            "TriggerGuard overrides (read)",
        )
        register(
            f"{PAGE_API_PREFIX}/overrides",
            self.post_overrides,
            ["POST"],
            "TriggerGuard overrides (write)",
        )
        register(
            f"{PAGE_API_PREFIX}/platforms",
            self.get_platforms,
            ["GET"],
            "TriggerGuard active platform instances",
        )
        register(
            f"{PAGE_API_PREFIX}/protocol_stats",
            self.get_protocol_stats,
            ["GET"],
            "TriggerGuard protocol stats (bot id / group list)",
        )
        register(
            f"{PAGE_API_PREFIX}/trigger_progress",
            self.get_trigger_progress,
            ["GET"],
            "TriggerGuard trigger progress (remaining/expected message count)",
        )

    async def get_platforms(self) -> dict[str, Any]:
        return _ok(self.plugin.get_platforms())

    async def get_protocol_stats(self) -> dict[str, Any]:
        platform_id = (request.args.get("platform_id") or "").strip()
        if not platform_id:
            return _error("缺少 platform_id 参数")
        return _ok(await self.plugin.get_protocol_stats(platform_id))

    async def get_trigger_progress(self) -> dict[str, Any]:
        platform_id = (request.args.get("platform_id") or "").strip()
        if not platform_id:
            return _error("缺少 platform_id 参数")
        return _ok(self.plugin.get_trigger_progress(platform_id))

    async def get_overrides(self) -> dict[str, Any]:
        self.plugin.maybe_reload()
        return _ok(self.plugin.serialize_overrides())

    async def post_overrides(self) -> dict[str, Any]:
        payload = await request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error("请求体必须是 JSON 对象")
        try:
            normalized = self.plugin.normalize_overrides(payload)
        except ValueError as exc:
            return _error(str(exc))
        self.plugin.save_overrides(normalized)
        return _ok(self.plugin.serialize_overrides())
