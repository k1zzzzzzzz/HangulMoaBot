# 한글 모아모아 봇 — 10×16 / Windows EXE 빌드판

## 가장 쉬운 방법: GitHub에서 EXE 만들기 (Python 설치 필요 없음)

이 패키지는 GitHub Actions가 **Windows 서버에서 자동으로 EXE를 빌드**하도록 설정되어 있습니다.

### 1. GitHub 계정
GitHub 계정이 없다면 먼저 계정을 만듭니다.

### 2. 새 저장소 만들기
GitHub에서 `New repository`를 눌러 아무 이름으로 저장소를 만듭니다.
예: `hangul-moa-bot`

Public/Private 어느 쪽도 가능합니다.

### 3. 이 폴더의 파일을 저장소에 업로드
아래 파일/폴더를 그대로 업로드합니다.

- `hangul_moa_bot.py`
- `requirements.txt`
- `HangulMoaBot_10x16.spec`
- `README.txt`
- `.github/workflows/build-windows-exe.yml`

### 4. Actions 실행
저장소의 `Actions` 탭으로 들어갑니다.

`Build Windows EXE` 워크플로가 실행되면 기다립니다.
완료되면 실행된 workflow를 클릭하고 `Artifacts`에서
`HangulMoaBot_10x16_EXE`를 다운로드합니다.

그 ZIP 안에 실제 Windows `.exe`가 들어 있습니다.

## EXE 실행

압축을 풀고:

`HangulMoaBot_10x16.exe`

를 실행합니다.

처음에는 보정 화면이 나옵니다.

1. 게임 보드의 **왼쪽 위 칸 중앙**에 마우스를 올리고 Enter
2. 게임 보드의 **오른쪽 아래 칸 중앙**에 마우스를 올리고 Enter

그 뒤 좌표가 저장되고 자동 플레이가 시작됩니다.

### 단축키

- F8: 일시정지 / 재개
- F9: 즉시 정지
- F6: 보드 좌표 재보정

### 먼저 테스트

EXE는 콘솔 창이 같이 뜨도록 만들었습니다.

현재 화면 인식만 확인하려면 명령행에서:

`HangulMoaBot_10x16.exe --debug`

첫 수 계산만 확인하려면:

`HangulMoaBot_10x16.exe --dry-run`

## 주의

- 메이플스토리 게임 창이 보이는 상태여야 합니다.
- 게임 창/화면 배율을 바꾸면 F6으로 다시 보정하세요.
- 처음에는 반드시 `--debug` 또는 `--dry-run`으로 확인하는 것을 권장합니다.
- 화면 인식 기반 프로토타입이므로 실제 클라이언트에서 조각/버튼 위치가 어긋나면
  해당 화면을 다시 보내면 좌표/인식 로직을 조정할 수 있습니다.

## 현재 구현

- 보드: 10×16
- 보유 조각 인식
- 회전/반전 탐색
- 3개 조각의 순서까지 고려하는 beam search
- 마우스 드래그 배치
- 줄 삭제를 고려한 점수 평가
- 점 찍기 스킬
- 바꿔 뽑기 스킬
- F8/F9/F6 단축키
