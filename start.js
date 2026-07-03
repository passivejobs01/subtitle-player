module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: { },
        message: [
          "python server.py --port {{port}}"
        ],
        on: [{
          // 서버가 출력하는 http URL을 캡처해 다음 스텝으로 전달
          "event": "/(http:\\/\\/[0-9.:]+)/",
          "done": true
        }]
      }
    },
    {
      // 캡처한 URL을 로컬 변수 url로 설정 → pinokio.js의 "플레이어 열기" 탭에 사용
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
