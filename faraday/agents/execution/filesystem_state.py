from __future__ import annotations

import asyncio


class FilesystemStateTracker:
    def __init__(self, runtime, log_fn):
        self.runtime = runtime
        self._log = log_fn
        self.root_files: list[str] = []
        self.data_files: list[str] = []
        self.plot_files: list[str] = []
        self.report_files: list[str] = []
        self.tex_files: list[str] = []
        self.webpage_files: list[str] = []

    async def refresh(self) -> bool:
        if not self.runtime.config.enable_sandbox:
            return False
        try:
            tasks = [
                asyncio.create_task(asyncio.to_thread(self._check_directory, "")),
                asyncio.create_task(asyncio.to_thread(self._check_directory, "data")),
                asyncio.create_task(asyncio.to_thread(self._check_directory, "plots")),
                asyncio.create_task(asyncio.to_thread(self._check_directory, "reports")),
                asyncio.create_task(asyncio.to_thread(self._check_directory, "tex")),
                asyncio.create_task(asyncio.to_thread(self._check_directory, "webpages")),
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=10.0,
            )
            self.root_files = self._to_lines(results[0], is_root=True)
            self.data_files = self._to_lines(results[1])
            self.plot_files = self._to_lines(results[2])
            self.report_files = self._to_lines(results[3])
            self.tex_files = self._to_lines(results[4])
            self.webpage_files = self._to_lines(results[5])
            return True
        except Exception as exc:
            self._log(f"Filesystem refresh failed: {exc}")
            self.root_files = []
            self.data_files = []
            self.plot_files = []
            self.report_files = []
            self.tex_files = []
            self.webpage_files = []
            return False

    def render_for_llm(self, include_summary: bool = True, include_details: bool = True) -> str:
        if not self.runtime.config.enable_sandbox:
            return "Sandbox filesystem: Not available (sandbox disabled)"
        lines: list[str] = []
        if include_summary:
            total = (
                len(self.root_files)
                + len(self.data_files)
                + len(self.plot_files)
                + len(self.report_files)
                + len(self.tex_files)
                + len(self.webpage_files)
            )
            lines.append(f"Sandbox Filesystem State ({total} total files):")
            lines.append("=" * 50)
        if include_details:
            self._append_dir_lines(lines, "./agent_outputs/ (root)", self.root_files)
            self._append_dir_lines(lines, "./agent_outputs/data/", self.data_files)
            self._append_dir_lines(lines, "./agent_outputs/plots/", self.plot_files)
            self._append_dir_lines(lines, "./agent_outputs/reports/", self.report_files)
            self._append_dir_lines(lines, "./agent_outputs/tex/", self.tex_files)
            self._append_dir_lines(lines, "./agent_outputs/webpages/", self.webpage_files)
        return "\n".join(lines) if lines else "Sandbox filesystem: No state information available"

    def render_compact(self) -> str:
        if not self.runtime.config.enable_sandbox:
            return "Sandbox: disabled"
        root_count = len(self.root_files)
        data_count = len(self.data_files)
        plot_count = len(self.plot_files)
        report_count = len(self.report_files)
        tex_count = len(self.tex_files)
        webpage_count = len(self.webpage_files)
        total = root_count + data_count + plot_count + report_count + tex_count + webpage_count
        if total == 0:
            return "Sandbox filesystem: empty"
        parts = []
        if root_count > 0:
            parts.append(f"{root_count} root files")
        if data_count > 0:
            parts.append(f"{data_count} data files")
        if plot_count > 0:
            parts.append(f"{plot_count} plot files")
        if report_count > 0:
            parts.append(f"{report_count} report files")
        if tex_count > 0:
            parts.append(f"{tex_count} tex files")
        if webpage_count > 0:
            parts.append(f"{webpage_count} webpage files")
        return f"Sandbox filesystem: {', '.join(parts)} ({total} total)"

    def check_root(self):
        return self._check_directory("")

    def check_data(self):
        return self._check_directory("data")

    def check_plots(self):
        return self._check_directory("plots")

    def check_storage_summary(self):
        text = []
        text.append("Storage Contents Summary:")
        text.append("-" * 50)
        text.append("Cloud Storage Files:")
        text.extend(f"\t{v}" for v in self._to_lines(self.check_root(), is_root=True))
        text.append("Data Files:")
        text.extend(f"\t{v}" for v in self._to_lines(self.check_data()))
        text.append("Plot Files:")
        text.extend(f"\t{v}" for v in self._to_lines(self.check_plots()))
        text.append("Report Files:")
        text.extend(f"\t{v}" for v in self._to_lines(self._check_directory("reports")))
        text.append("TeX Files:")
        text.extend(f"\t{v}" for v in self._to_lines(self._check_directory("tex")))
        text.append("-" * 50)
        return "\n".join(text)

    def _check_directory(self, subdir: str = "") -> str:
        root = self.runtime.sandbox_outputs_root(cloud_storage_available=True)
        path = root + (f"/{subdir}" if subdir else "")
        try:
            sb = self.runtime.get_sandbox()
            cmd = f'ls -1 "{path}" 2>/dev/null || true'
            proc = sb.exec("bash", "-lc", cmd)
            proc.wait()
            if proc.stdout:
                raw = proc.stdout.read()
                return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            return ""
        except Exception as exc:
            self._log(f"Storage listing failed for {subdir or 'root'}: {exc}")
            return ""

    def _to_lines(self, value, is_root: bool = False) -> list[str]:
        if isinstance(value, Exception) or value is None:
            return []
        lines = [x.strip() for x in str(value).split("\n") if x.strip()]
        if is_root:
            lines = [
                x
                for x in lines
                if x not in [".modal_code_vars", "data", "plots", "reports", "tex", "webpages"]
            ]
        return lines

    def _append_dir_lines(self, lines: list[str], header: str, values: list[str]) -> None:
        if values:
            lines.append(f"📁 {header}:")
            for file_name in sorted(values):
                lines.append(f"  - {file_name}")
        else:
            lines.append(f"📁 {header}: empty")
