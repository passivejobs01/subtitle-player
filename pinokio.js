module.exports = {
  version: "7.0",
  title: "자막 학습 플레이어",
  description: "유튜브·로컬 영상에 원어+한글 이중 자막을 만들어 외국어를 공부하는 로컬 플레이어 — 잇츠매거진",
  menu: async (kernel, info) => {
    let installed = info.exists("app/env")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }
    if (running.install) {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Installing", href: "install.js" }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [
            // popout: Pinokio 웹뷰 대신 외부 브라우저로 연다(웹뷰에선 YouTube 임베드가 막히므로 브라우저 필수)
            { default: true, icon: "fa-solid fa-rocket", text: "브라우저에서 플레이어 열기", href: local.url, popout: true },
            { icon: "fa-solid fa-terminal", text: "Terminal (주소 확인)", href: "start.js" }
          ]
        } else {
          return [{ default: true, icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" }]
        }
      } else if (running.update) {
        return [{ default: true, icon: "fa-solid fa-terminal", text: "Updating", href: "update.js" }]
      } else if (running.reset) {
        return [{ default: true, icon: "fa-solid fa-terminal", text: "Resetting", href: "reset.js" }]
      } else {
        return [
          { default: true, icon: "fa-solid fa-power-off", text: "Start", href: "start.js" },
          { icon: "fa-solid fa-plug", text: "Update", href: "update.js" },
          { icon: "fa-solid fa-plug", text: "Install", href: "install.js" },
          { icon: "fa-regular fa-circle-xmark", text: "<div><strong>Reset</strong><div>설치 초기화</div></div>", href: "reset.js", confirm: "설치를 초기화할까요?" }
        ]
      }
    } else {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Install", href: "install.js" }]
    }
  }
}
